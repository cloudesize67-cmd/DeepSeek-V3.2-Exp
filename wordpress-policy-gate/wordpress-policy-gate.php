<?php
/**
 * Plugin Name: WordPress Policy Gate
 * Plugin URI:  https://github.com/example/wordpress-policy-gate
 * Description: Intercepts AI agent and admin actions, enforces configurable policies
 *              (e.g. no-bulk-delete), and records every decision to a
 *              Merkle-tree authenticated, cryptographically verifiable audit log.
 * Version:     1.0.0
 * Author:      Policy Gate Contributors
 * License:     GPL-2.0-or-later
 * Text Domain: wpg
 *
 * @package WPG
 */

declare(strict_types=1);

defined('ABSPATH') || exit;

define('WPG_VERSION', '1.0.0');
define('WPG_DIR',     plugin_dir_path(__FILE__));
define('WPG_URL',     plugin_dir_url(__FILE__));

// ---------------------------------------------------------------------------
// Autoloader
// ---------------------------------------------------------------------------
spl_autoload_register(static function (string $class): void {
    if (strpos($class, 'WPG\\') !== 0) {
        return;
    }
    $rel  = str_replace(['WPG\\', '\\'], ['', '/'], $class);
    $file = WPG_DIR . 'src/' . $rel . '.php';
    if (file_exists($file)) {
        require_once $file;
    }
});

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
add_action('plugins_loaded', static function (): void {
    WPG\Policy\PolicyGate::instance()->register_hooks();
    WPG\Api\RestEndpoints::instance()->register();
});

register_activation_hook(__FILE__, ['WPG\Installer', 'activate']);
register_deactivation_hook(__FILE__, ['WPG\Installer', 'deactivate']);
