<?php
/**
 * Central policy gate.
 *
 * Flow:
 *   1. Intercept an incoming action request.
 *   2. Evaluate every enabled policy against it.
 *   3. If any policy DENYs → block the action, log, alert.
 *   4. If all policies pass  → allow, log.
 *
 * @package WPG\Policy
 */

declare(strict_types=1);

namespace WPG\Policy;

use WPG\Audit\MerkleAuditLog;
use WPG\Alert\AlertDispatcher;

defined('ABSPATH') || exit;

/**
 * @phpstan-type ActionContext array{
 *   action: string,
 *   count:  int,
 *   actor:  array{id: int, roles: string[]},
 *   meta:   array<string, mixed>
 * }
 */
final class PolicyGate
{
    private static ?self $instance = null;

    /** @var array<string, PolicyRule> */
    private array $rules = [];

    private MerkleAuditLog  $audit;
    private AlertDispatcher $alerts;

    private function __construct()
    {
        $this->audit  = new MerkleAuditLog();
        $this->alerts = new AlertDispatcher();

        // Register built-in rules.
        $this->register_rule(new Rules\NoBulkDeleteRule());
    }

    public static function instance(): self
    {
        if (self::$instance === null) {
            self::$instance = new self();
        }
        return self::$instance;
    }

    // -----------------------------------------------------------------------
    // Hook registration
    // -----------------------------------------------------------------------

    public function register_hooks(): void
    {
        // WP REST API: intercept bulk-trash / bulk-delete from the Block Editor.
        add_filter('rest_pre_dispatch', [$this, 'intercept_rest_request'], 1, 3);

        // Classic admin: bulk actions on the Posts list table.
        add_filter('handle_bulk_actions-edit-post', [$this, 'intercept_bulk_action'], 1, 3);

        // Programmatic / AI-agent entry point (custom hook).
        add_filter('wpg_evaluate_action', [$this, 'evaluate'], 1, 2);
    }

    // -----------------------------------------------------------------------
    // REST API interceptor
    // -----------------------------------------------------------------------

    /**
     * @param mixed            $result
     * @param \WP_REST_Server  $server
     * @param \WP_REST_Request $request
     * @return mixed  WP_Error on DENY, original $result otherwise.
     */
    public function intercept_rest_request($result, $server, $request)
    {
        // Only inspect batch / delete endpoints.
        $route  = $request->get_route();
        $method = $request->get_method();

        // e.g. DELETE /wp/v2/posts  or  POST /wp/v2/posts with force=true
        if (!preg_match('#^/wp/v2/(posts|pages|media|comments)#', $route)) {
            return $result;
        }
        if ($method !== 'DELETE' && !($method === 'POST' && $request->get_param('force'))) {
            return $result;
        }

        $ids = (array) ($request->get_param('id') ?? []);
        if (empty($ids)) {
            return $result;
        }

        $verdict = $this->evaluate(
            null,
            $this->build_context('core/delete-posts', count($ids), $request->get_param('id'))
        );

        if ($verdict['verdict'] === 'DENIED') {
            return new \WP_Error(
                'wpg_policy_denied',
                $verdict['message'],
                ['status' => 403, 'policy' => $verdict['policy']]
            );
        }

        return $result;
    }

    // -----------------------------------------------------------------------
    // Classic-admin bulk-action interceptor
    // -----------------------------------------------------------------------

    /**
     * @param string   $redirect_url
     * @param string   $action
     * @param int[]    $post_ids
     * @return string  Original redirect_url (action was allowed) or a redirect
     *                 to the list table with an error notice appended.
     */
    public function intercept_bulk_action(string $redirect_url, string $action, array $post_ids): string
    {
        if (!in_array($action, ['delete', 'trash'], true)) {
            return $redirect_url;
        }

        $wpg_action = ($action === 'delete') ? 'core/delete-posts' : 'core/trash-posts';
        $verdict    = $this->evaluate(
            null,
            $this->build_context($wpg_action, count($post_ids), $post_ids)
        );

        if ($verdict['verdict'] === 'DENIED') {
            // Surface the denial as an admin notice.
            set_transient('wpg_bulk_action_denied_' . get_current_user_id(), $verdict['message'], 60);
            return add_query_arg('wpg_denied', '1', remove_query_arg(['deleted', 'trashed'], $redirect_url));
        }

        return $redirect_url;
    }

    // -----------------------------------------------------------------------
    // Core evaluation engine
    // -----------------------------------------------------------------------

    /**
     * Evaluate an action against all active policies.
     *
     * Can be called directly or via `apply_filters('wpg_evaluate_action', null, $ctx)`.
     *
     * @param  mixed                  $passthrough  Ignored (filter chaining).
     * @param  array<string, mixed>   $context      ActionContext shape.
     * @return array{verdict: string, policy: string, message: string, audit_id: int}
     */
    public function evaluate($passthrough, array $context): array
    {
        $context = $this->normalize_context($context);

        $policies = (array) get_option('wpg_policies', []);
        $denial   = null;

        foreach ($policies as $policy_cfg) {
            if (empty($policy_cfg['enabled'])) {
                continue;
            }

            $rule_name = $policy_cfg['rule'] ?? '';
            if (!isset($this->rules[$rule_name])) {
                continue;
            }

            if (!$this->action_matches($context['action'], (array) ($policy_cfg['actions'] ?? []))) {
                continue;
            }

            $result = $this->rules[$rule_name]->evaluate($context, $policy_cfg);

            if ($result['verdict'] === 'DENIED') {
                $denial = array_merge($result, ['policy_name' => $policy_cfg['name']]);
                break; // First denial wins.
            }
        }

        $verdict     = $denial ? 'DENIED' : 'ALLOWED';
        $policy_name = $denial['policy_name'] ?? '';

        // Write immutable audit record.
        $audit_id = $this->audit->record([
            'action_id'   => wp_generate_uuid4(),
            'action_name' => $context['action'],
            'actor_id'    => $context['actor']['id'],
            'actor_role'  => implode(',', $context['actor']['roles']),
            'verdict'     => $verdict,
            'policy_name' => $policy_name,
            'item_count'  => $context['count'],
            'context'     => $context,
        ]);

        if ($verdict === 'DENIED') {
            $message = $denial['message'] ?? sprintf(
                'Bulk deletion of %d+ items requires human approval.',
                $denial['limit'] ?? 10
            );
            $this->alerts->dispatch($context, $denial, $audit_id);
        } else {
            $message = 'Action allowed.';
        }

        return [
            'verdict'  => $verdict,
            'policy'   => $policy_name,
            'message'  => $message,
            'audit_id' => $audit_id,
        ];
    }

    // -----------------------------------------------------------------------
    // Rule registration
    // -----------------------------------------------------------------------

    public function register_rule(PolicyRule $rule): void
    {
        $this->rules[$rule->name()] = $rule;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /**
     * @param  string  $action
     * @param  mixed   $item_ids
     * @return array<string, mixed>
     */
    private function build_context(string $action, int $count, $item_ids = []): array
    {
        $user  = wp_get_current_user();
        return [
            'action' => $action,
            'count'  => $count,
            'ids'    => (array) $item_ids,
            'actor'  => [
                'id'    => (int) $user->ID,
                'roles' => (array) $user->roles,
            ],
            'meta'   => [],
        ];
    }

    /**
     * Fill in missing keys with safe defaults.
     *
     * @param  array<string, mixed> $ctx
     * @return array<string, mixed>
     */
    private function normalize_context(array $ctx): array
    {
        return array_merge(
            [
                'action' => '',
                'count'  => 0,
                'ids'    => [],
                'actor'  => ['id' => 0, 'roles' => []],
                'meta'   => [],
            ],
            $ctx
        );
    }

    /**
     * Returns true when $action matches any of the glob-style $patterns.
     *
     * @param string[] $patterns
     */
    private function action_matches(string $action, array $patterns): bool
    {
        foreach ($patterns as $pattern) {
            // Convert glob wildcards to regex.
            $regex = '#^' . str_replace(['\*', '\?'], ['.*', '.'], preg_quote($pattern, '#')) . '$#';
            if (preg_match($regex, $action)) {
                return true;
            }
        }
        return false;
    }
}
