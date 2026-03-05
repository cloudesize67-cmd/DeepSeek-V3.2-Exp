<?php
/**
 * Handles plugin activation / deactivation (DB table creation, default options).
 *
 * @package WPG
 */

declare(strict_types=1);

namespace WPG;

defined('ABSPATH') || exit;

class Installer
{
    /** Database table that stores audit-log entries. */
    public const TABLE = 'wpg_audit_log';

    public static function activate(): void
    {
        self::create_table();
        self::seed_default_options();
        flush_rewrite_rules();
    }

    public static function deactivate(): void
    {
        flush_rewrite_rules();
    }

    // -----------------------------------------------------------------------

    private static function create_table(): void
    {
        global $wpdb;

        $table   = $wpdb->prefix . self::TABLE;
        $charset = $wpdb->get_charset_collate();

        $sql = "CREATE TABLE IF NOT EXISTS {$table} (
            id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
            created_at    DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
            action_id     VARCHAR(64)     NOT NULL,
            action_name   VARCHAR(255)    NOT NULL,
            actor_id      BIGINT UNSIGNED NOT NULL DEFAULT 0,
            actor_role    VARCHAR(64)     NOT NULL DEFAULT '',
            verdict       ENUM('ALLOWED','DENIED') NOT NULL,
            policy_name   VARCHAR(128)    NOT NULL DEFAULT '',
            item_count    INT UNSIGNED    NOT NULL DEFAULT 0,
            context_json  LONGTEXT        NOT NULL,
            leaf_hash     VARCHAR(64)     NOT NULL,
            tree_root     VARCHAR(64)     NOT NULL,
            PRIMARY KEY (id),
            KEY idx_created_at  (created_at),
            KEY idx_action_name (action_name),
            KEY idx_verdict     (verdict)
        ) {$charset};";

        require_once ABSPATH . 'wp-admin/includes/upgrade.php';
        dbDelta($sql);

        update_option('wpg_db_version', WPG_VERSION);
    }

    private static function seed_default_options(): void
    {
        // Only write defaults if they don't exist yet.
        add_option('wpg_policies', self::default_policies());
        add_option('wpg_alert_webhook_url', '');
        add_option('wpg_alert_email', get_option('admin_email'));
    }

    /**
     * Default policy configuration.
     *
     * Each policy is an associative array with at least:
     *   - name      (string)
     *   - actions   (string[])  glob-style patterns matched against action names
     *   - rule      (string)    one of: no-bulk-delete | rate-limit | …
     *   - limit     (int)       semantics depend on rule
     *   - enabled   (bool)
     *
     * @return array<int, array<string, mixed>>
     */
    public static function default_policies(): array
    {
        return [
            [
                'name'    => 'no-bulk-delete',
                'actions' => ['core/delete-posts', 'core/delete-*'],
                'rule'    => 'no-bulk-delete',
                'limit'   => 10,
                'enabled' => true,
            ],
        ];
    }
}
