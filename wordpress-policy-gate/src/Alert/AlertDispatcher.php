<?php
/**
 * Dispatches denial alerts via webhook and/or email.
 *
 * When a policy denies an action:
 *   1. POST a structured JSON payload to the configured webhook URL.
 *   2. Send a formatted HTML email to the site admin (or configured address).
 *
 * Both channels are fire-and-forget; failures are logged to error_log but
 * do NOT bubble up to the caller (the gate verdict is already decided).
 *
 * @package WPG\Alert
 */

declare(strict_types=1);

namespace WPG\Alert;

defined('ABSPATH') || exit;

final class AlertDispatcher
{
    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * Fire all configured alert channels.
     *
     * @param array<string, mixed> $context    ActionContext for the denied action.
     * @param array<string, mixed> $denial     Denial details from the policy rule.
     * @param int                  $audit_id   Row ID in the audit log.
     */
    public function dispatch(array $context, array $denial, int $audit_id): void
    {
        $payload = $this->build_payload($context, $denial, $audit_id);

        $webhook_url = (string) get_option('wpg_alert_webhook_url', '');
        if ($webhook_url !== '') {
            $this->send_webhook($webhook_url, $payload);
        }

        $email = (string) get_option('wpg_alert_email', get_option('admin_email'));
        if ($email !== '') {
            $this->send_email($email, $payload);
        }

        /**
         * Hook: 'wpg_alert_dispatched'
         *
         * Allows third-party code to add additional alert channels (Slack, PagerDuty, …).
         *
         * @param array<string, mixed> $payload  Structured alert payload.
         */
        do_action('wpg_alert_dispatched', $payload);
    }

    // -----------------------------------------------------------------------
    // Payload builder
    // -----------------------------------------------------------------------

    /**
     * @param  array<string, mixed> $context
     * @param  array<string, mixed> $denial
     * @return array<string, mixed>
     */
    private function build_payload(array $context, array $denial, int $audit_id): array
    {
        return [
            'event'      => 'wpg.policy.denied',
            'timestamp'  => gmdate('c'),
            'site_url'   => get_site_url(),
            'audit_id'   => $audit_id,
            'action'     => [
                'name'       => $context['action'] ?? '',
                'item_count' => $context['count']  ?? 0,
                'item_ids'   => array_slice((array) ($context['ids'] ?? []), 0, 25), // cap preview
            ],
            'actor'      => [
                'wp_user_id' => $context['actor']['id']    ?? 0,
                'roles'      => $context['actor']['roles'] ?? [],
                'login'      => $this->get_user_login((int) ($context['actor']['id'] ?? 0)),
            ],
            'policy'     => [
                'name'    => $denial['policy_name'] ?? '',
                'limit'   => $denial['limit']       ?? 0,
                'verdict' => 'DENIED',
                'message' => $denial['message']     ?? '',
            ],
        ];
    }

    // -----------------------------------------------------------------------
    // Webhook
    // -----------------------------------------------------------------------

    /** @param array<string, mixed> $payload */
    private function send_webhook(string $url, array $payload): void
    {
        $body     = (string) wp_json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
        $hmac_key = (string) get_option('wpg_webhook_secret', wp_salt('auth'));
        $signature = 'sha256=' . hash_hmac('sha256', $body, $hmac_key);

        $response = wp_remote_post($url, [
            'timeout'     => 5,
            'redirection' => 2,
            'headers'     => [
                'Content-Type'     => 'application/json',
                'X-WPG-Signature'  => $signature,
                'X-WPG-Event'      => 'policy.denied',
                'X-WPG-Audit-ID'   => (string) $payload['audit_id'],
            ],
            'body' => $body,
        ]);

        if (is_wp_error($response)) {
            error_log('[WPG] Webhook delivery failed: ' . $response->get_error_message());
        }
    }

    // -----------------------------------------------------------------------
    // Email
    // -----------------------------------------------------------------------

    /** @param array<string, mixed> $payload */
    private function send_email(string $to, array $payload): void
    {
        $site    = get_bloginfo('name');
        $subject = sprintf('[%s] Policy Gate Alert: bulk delete blocked (audit #%d)', $site, $payload['audit_id']);

        $body = $this->render_email_html($payload);

        $headers = ['Content-Type: text/html; charset=UTF-8'];

        $sent = wp_mail($to, $subject, $body, $headers);
        if (!$sent) {
            error_log('[WPG] Admin email delivery failed for audit #' . $payload['audit_id']);
        }
    }

    /** @param array<string, mixed> $payload */
    private function render_email_html(array $payload): string
    {
        $actor   = $payload['actor'];
        $action  = $payload['action'];
        $policy  = $payload['policy'];
        $site    = esc_html(get_bloginfo('name'));
        $ts      = esc_html($payload['timestamp']);
        $audit   = (int) $payload['audit_id'];

        ob_start();
        ?>
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="font-family:Arial,sans-serif;color:#333;max-width:600px;margin:0 auto">
  <h2 style="color:#b91c1c">&#9888; Policy Gate — Action Denied</h2>
  <p>A bulk-delete action was <strong>blocked</strong> by the WordPress Policy Gate on <strong><?php echo $site; ?></strong>.</p>

  <table style="border-collapse:collapse;width:100%">
    <tr><th style="text-align:left;padding:6px 8px;background:#f3f4f6;width:35%">Timestamp</th>
        <td style="padding:6px 8px;border-bottom:1px solid #e5e7eb"><?php echo $ts; ?></td></tr>
    <tr><th style="text-align:left;padding:6px 8px;background:#f3f4f6">Audit ID</th>
        <td style="padding:6px 8px;border-bottom:1px solid #e5e7eb">#<?php echo $audit; ?></td></tr>
    <tr><th style="text-align:left;padding:6px 8px;background:#f3f4f6">Actor</th>
        <td style="padding:6px 8px;border-bottom:1px solid #e5e7eb">
          <?php echo esc_html($actor['login']); ?> (ID <?php echo (int) $actor['wp_user_id']; ?>,
          roles: <?php echo esc_html(implode(', ', (array) $actor['roles'])); ?>)
        </td></tr>
    <tr><th style="text-align:left;padding:6px 8px;background:#f3f4f6">Action</th>
        <td style="padding:6px 8px;border-bottom:1px solid #e5e7eb"><?php echo esc_html($action['name']); ?></td></tr>
    <tr><th style="text-align:left;padding:6px 8px;background:#f3f4f6">Items targeted</th>
        <td style="padding:6px 8px;border-bottom:1px solid #e5e7eb"><?php echo (int) $action['item_count']; ?></td></tr>
    <tr><th style="text-align:left;padding:6px 8px;background:#f3f4f6">Policy</th>
        <td style="padding:6px 8px;border-bottom:1px solid #e5e7eb"><?php echo esc_html($policy['name']); ?>
          (limit: <?php echo (int) $policy['limit']; ?>)</td></tr>
    <tr><th style="text-align:left;padding:6px 8px;background:#f3f4f6">Message</th>
        <td style="padding:6px 8px"><?php echo esc_html($policy['message']); ?></td></tr>
  </table>

  <p style="margin-top:20px;font-size:12px;color:#6b7280">
    This alert was generated automatically by the WordPress Policy Gate plugin.
    Audit record #<?php echo $audit; ?> is cryptographically verifiable in the site's Merkle audit log.
  </p>
</body>
</html>
        <?php
        return (string) ob_get_clean();
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private function get_user_login(int $user_id): string
    {
        if ($user_id <= 0) {
            return 'unknown';
        }
        $user = get_userdata($user_id);
        return $user ? $user->user_login : "user_{$user_id}";
    }
}
