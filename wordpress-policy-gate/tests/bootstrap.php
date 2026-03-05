<?php
/**
 * PHPUnit bootstrap — no WordPress environment needed.
 *
 * Only define the ABSPATH constant so plugin files that guard with
 * `defined('ABSPATH') || exit` don't bail out.
 */

declare(strict_types=1);

if (!defined('ABSPATH')) {
    define('ABSPATH', dirname(__DIR__) . '/');
}

// If Composer autoloader exists (CI), use it; otherwise fall back to the
// manual requires inside each test file.
$autoload = dirname(__DIR__) . '/vendor/autoload.php';
if (file_exists($autoload)) {
    require_once $autoload;
}
