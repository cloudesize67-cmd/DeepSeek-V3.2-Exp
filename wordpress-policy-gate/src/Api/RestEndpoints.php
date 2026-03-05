<?php
/**
 * REST API endpoints exposed by the Policy Gate.
 *
 * Namespace: wpg/v1
 *
 * Endpoints:
 *   POST /wpg/v1/evaluate          – Evaluate an action (AI-agent entry point)
 *   GET  /wpg/v1/audit             – List recent audit records
 *   GET  /wpg/v1/audit/verify      – Verify Merkle-tree integrity
 *   GET  /wpg/v1/policies          – List configured policies
 *   PUT  /wpg/v1/policies          – Update policies (admin only)
 *
 * @package WPG\Api
 */

declare(strict_types=1);

namespace WPG\Api;

use WPG\Policy\PolicyGate;
use WPG\Audit\MerkleAuditLog;

defined('ABSPATH') || exit;

final class RestEndpoints
{
    private const NAMESPACE = 'wpg/v1';

    private static ?self $instance = null;

    public static function instance(): self
    {
        if (self::$instance === null) {
            self::$instance = new self();
        }
        return self::$instance;
    }

    public function register(): void
    {
        add_action('rest_api_init', [$this, 'register_routes']);
    }

    public function register_routes(): void
    {
        // ----------------------------------------------------------------
        // POST /wpg/v1/evaluate
        // ----------------------------------------------------------------
        register_rest_route(self::NAMESPACE, '/evaluate', [
            'methods'             => \WP_REST_Server::CREATABLE,
            'callback'            => [$this, 'evaluate'],
            'permission_callback' => [$this, 'require_authenticated'],
            'args'                => [
                'action' => [
                    'required'          => true,
                    'type'              => 'string',
                    'sanitize_callback' => 'sanitize_text_field',
                    'description'       => 'Action identifier, e.g. core/delete-posts',
                ],
                'count'  => [
                    'required'          => true,
                    'type'              => 'integer',
                    'minimum'           => 0,
                    'description'       => 'Number of items targeted.',
                ],
                'ids'    => [
                    'type'        => 'array',
                    'items'       => ['type' => 'integer'],
                    'default'     => [],
                    'description' => 'Optional list of item IDs.',
                ],
                'meta'   => [
                    'type'    => 'object',
                    'default' => [],
                ],
            ],
        ]);

        // ----------------------------------------------------------------
        // GET /wpg/v1/audit
        // ----------------------------------------------------------------
        register_rest_route(self::NAMESPACE, '/audit', [
            'methods'             => \WP_REST_Server::READABLE,
            'callback'            => [$this, 'list_audit'],
            'permission_callback' => [$this, 'require_manage_options'],
            'args'                => [
                'limit' => [
                    'type'    => 'integer',
                    'default' => 50,
                    'minimum' => 1,
                    'maximum' => 500,
                ],
            ],
        ]);

        // ----------------------------------------------------------------
        // GET /wpg/v1/audit/verify
        // ----------------------------------------------------------------
        register_rest_route(self::NAMESPACE, '/audit/verify', [
            'methods'             => \WP_REST_Server::READABLE,
            'callback'            => [$this, 'verify_audit'],
            'permission_callback' => [$this, 'require_manage_options'],
        ]);

        // ----------------------------------------------------------------
        // GET /wpg/v1/policies
        // ----------------------------------------------------------------
        register_rest_route(self::NAMESPACE, '/policies', [
            [
                'methods'             => \WP_REST_Server::READABLE,
                'callback'            => [$this, 'get_policies'],
                'permission_callback' => [$this, 'require_manage_options'],
            ],
            [
                'methods'             => \WP_REST_Server::EDITABLE,
                'callback'            => [$this, 'update_policies'],
                'permission_callback' => [$this, 'require_manage_options'],
                'args'                => [
                    'policies' => [
                        'required' => true,
                        'type'     => 'array',
                    ],
                ],
            ],
        ]);
    }

    // -----------------------------------------------------------------------
    // Handlers
    // -----------------------------------------------------------------------

    public function evaluate(\WP_REST_Request $request): \WP_REST_Response
    {
        $user = wp_get_current_user();

        $context = [
            'action' => $request->get_param('action'),
            'count'  => (int) $request->get_param('count'),
            'ids'    => (array) ($request->get_param('ids') ?? []),
            'actor'  => [
                'id'    => (int) $user->ID,
                'roles' => (array) $user->roles,
            ],
            'meta'   => (array) ($request->get_param('meta') ?? []),
        ];

        $result = PolicyGate::instance()->evaluate(null, $context);

        $status = $result['verdict'] === 'DENIED' ? 403 : 200;
        return new \WP_REST_Response($result, $status);
    }

    public function list_audit(\WP_REST_Request $request): \WP_REST_Response
    {
        $log     = new MerkleAuditLog();
        $records = $log->recent((int) $request->get_param('limit'));
        return new \WP_REST_Response(['records' => $records, 'count' => count($records)], 200);
    }

    public function verify_audit(\WP_REST_Request $request): \WP_REST_Response
    {
        $log    = new MerkleAuditLog();
        $result = $log->verify();
        $status = $result['valid'] ? 200 : 409;
        return new \WP_REST_Response($result, $status);
    }

    public function get_policies(\WP_REST_Request $request): \WP_REST_Response
    {
        return new \WP_REST_Response(['policies' => get_option('wpg_policies', [])], 200);
    }

    public function update_policies(\WP_REST_Request $request): \WP_REST_Response
    {
        $raw      = (array) $request->get_param('policies');
        $sanitized = $this->sanitize_policies($raw);

        update_option('wpg_policies', $sanitized);
        return new \WP_REST_Response(['policies' => $sanitized, 'updated' => true], 200);
    }

    // -----------------------------------------------------------------------
    // Permission callbacks
    // -----------------------------------------------------------------------

    public function require_authenticated(): bool|\WP_Error
    {
        if (!is_user_logged_in()) {
            return new \WP_Error('rest_forbidden', 'Authentication required.', ['status' => 401]);
        }
        return true;
    }

    public function require_manage_options(): bool|\WP_Error
    {
        if (!current_user_can('manage_options')) {
            return new \WP_Error('rest_forbidden', 'Insufficient permissions.', ['status' => 403]);
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /**
     * @param  array<int, mixed>            $raw
     * @return array<int, array<string, mixed>>
     */
    private function sanitize_policies(array $raw): array
    {
        $out = [];
        foreach ($raw as $p) {
            if (!is_array($p)) {
                continue;
            }
            $out[] = [
                'name'    => sanitize_text_field($p['name']    ?? ''),
                'actions' => array_map('sanitize_text_field', (array) ($p['actions'] ?? [])),
                'rule'    => sanitize_text_field($p['rule']    ?? ''),
                'limit'   => max(1, (int) ($p['limit'] ?? 10)),
                'enabled' => (bool) ($p['enabled'] ?? true),
            ];
        }
        return $out;
    }
}
