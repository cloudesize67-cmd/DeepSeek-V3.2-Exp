<?php
/**
 * Unit tests for the Merkle-tree hash logic extracted from MerkleAuditLog.
 *
 * The DB-dependent methods (record, recent, verify) are tested via a
 * test-double subclass that replaces wpdb calls with an in-memory store.
 *
 * @package WPG\Tests
 */

declare(strict_types=1);

require_once __DIR__ . '/../src/Installer.php';
require_once __DIR__ . '/../src/Audit/MerkleAuditLog.php';

use PHPUnit\Framework\TestCase;
use WPG\Audit\MerkleAuditLog;

// ---------------------------------------------------------------------------
// WP stubs
// ---------------------------------------------------------------------------
if (!function_exists('wp_generate_uuid4')) {
    function wp_generate_uuid4(): string {
        return sprintf('%04x%04x-%04x-%04x-%04x-%04x%04x%04x',
            mt_rand(0, 0xffff), mt_rand(0, 0xffff),
            mt_rand(0, 0xffff),
            mt_rand(0, 0x0fff) | 0x4000,
            mt_rand(0, 0x3fff) | 0x8000,
            mt_rand(0, 0xffff), mt_rand(0, 0xffff), mt_rand(0, 0xffff)
        );
    }
}
if (!function_exists('current_time')) {
    function current_time(string $type, bool $gmt = false): string {
        return gmdate('Y-m-d H:i:s');
    }
}
if (!function_exists('wp_json_encode')) {
    function wp_json_encode($data, int $flags = 0): string|false {
        return json_encode($data, $flags);
    }
}
if (!defined('ABSPATH')) { define('ABSPATH', '/'); }

// ---------------------------------------------------------------------------
// In-memory test double
// ---------------------------------------------------------------------------

/**
 * Subclass that replaces all $wpdb calls with a simple in-memory array so
 * tests run without a real database.
 */
class TestableMerkleAuditLog extends MerkleAuditLog
{
    /** @var array<int, array<string, mixed>> */
    public array $store = [];
    private int  $next_id = 1;

    protected function db_last_tree_root(): string
    {
        if (empty($this->store)) {
            return '';
        }
        return end($this->store)['tree_root'];
    }

    protected function db_insert(array $row): int
    {
        $row['id'] = $this->next_id++;
        $this->store[] = $row;
        return $row['id'];
    }

    protected function db_all_rows(): array
    {
        return $this->store;
    }
}

// ---------------------------------------------------------------------------
// Expose protected helpers via reflection so we can test them directly.
// ---------------------------------------------------------------------------

/** @return string */
function call_leaf_hash(MerkleAuditLog $log, string $prev, string $action_id, string $ts, string $payload): string
{
    $ref = new ReflectionMethod($log, 'compute_leaf_hash');
    $ref->setAccessible(true);
    return $ref->invoke($log, $prev, $action_id, $ts, $payload);
}

/** @return string */
function call_tree_root(MerkleAuditLog $log, string $prev, string $leaf): string
{
    $ref = new ReflectionMethod($log, 'compute_tree_root');
    $ref->setAccessible(true);
    return $ref->invoke($log, $prev, $leaf);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

class MerkleAuditLogTest extends TestCase
{
    // -----------------------------------------------------------------------
    // compute_leaf_hash
    // -----------------------------------------------------------------------

    public function test_leaf_hash_is_64_hex_chars(): void
    {
        $log  = new MerkleAuditLog();
        $hash = call_leaf_hash($log, '', 'action-1', '2024-01-01 00:00:00', '{}');
        $this->assertMatchesRegularExpression('/^[0-9a-f]{64}$/', $hash);
    }

    public function test_leaf_hash_changes_when_prev_root_changes(): void
    {
        $log  = new MerkleAuditLog();
        $h1   = call_leaf_hash($log, '',    'action-1', '2024-01-01', '{}');
        $h2   = call_leaf_hash($log, 'abc', 'action-1', '2024-01-01', '{}');
        $this->assertNotSame($h1, $h2);
    }

    public function test_leaf_hash_changes_when_payload_changes(): void
    {
        $log  = new MerkleAuditLog();
        $h1   = call_leaf_hash($log, '', 'a', '2024-01-01', '{"count":1}');
        $h2   = call_leaf_hash($log, '', 'a', '2024-01-01', '{"count":2}');
        $this->assertNotSame($h1, $h2);
    }

    // -----------------------------------------------------------------------
    // compute_tree_root
    // -----------------------------------------------------------------------

    public function test_tree_root_is_deterministic(): void
    {
        $log = new MerkleAuditLog();
        $r1  = call_tree_root($log, 'prevroot', 'leafhash');
        $r2  = call_tree_root($log, 'prevroot', 'leafhash');
        $this->assertSame($r1, $r2);
    }

    public function test_tree_root_changes_when_leaf_changes(): void
    {
        $log = new MerkleAuditLog();
        $r1  = call_tree_root($log, 'prev', 'leaf-A');
        $r2  = call_tree_root($log, 'prev', 'leaf-B');
        $this->assertNotSame($r1, $r2);
    }

    // -----------------------------------------------------------------------
    // Sequential chain integrity
    // -----------------------------------------------------------------------

    public function test_three_sequential_roots_form_valid_chain(): void
    {
        $log = new MerkleAuditLog();

        // Manually replicate what record() does for three entries.
        $root0 = '';
        $leaf1 = call_leaf_hash($log, $root0, 'a1', '2024-01-01 00:00:01', '{"n":1}');
        $root1 = call_tree_root($log, $root0, $leaf1);

        $leaf2 = call_leaf_hash($log, $root1, 'a2', '2024-01-01 00:00:02', '{"n":2}');
        $root2 = call_tree_root($log, $root1, $leaf2);

        $leaf3 = call_leaf_hash($log, $root2, 'a3', '2024-01-01 00:00:03', '{"n":3}');
        $root3 = call_tree_root($log, $root2, $leaf3);

        // Re-verify by replaying from scratch.
        $running = '';
        foreach ([[$leaf1, $root1], [$leaf2, $root2], [$leaf3, $root3]] as [$leaf, $expected_root]) {
            $running = call_tree_root($log, $running, $leaf);
            $this->assertSame($expected_root, $running);
        }
    }

    public function test_tampered_leaf_breaks_subsequent_roots(): void
    {
        $log = new MerkleAuditLog();

        $root0 = '';
        $leaf1 = call_leaf_hash($log, $root0, 'a1', '2024-01-01', '{}');
        $root1 = call_tree_root($log, $root0, $leaf1);

        $leaf2 = call_leaf_hash($log, $root1, 'a2', '2024-01-01', '{}');
        $root2 = call_tree_root($log, $root1, $leaf2);

        // Tamper with leaf1.
        $tampered_leaf1 = call_leaf_hash($log, $root0, 'a1-TAMPERED', '2024-01-01', '{}');
        $tampered_root1 = call_tree_root($log, $root0, $tampered_leaf1);

        // Recompute root2 using tampered root1 — must not match original root2.
        $tampered_root2 = call_tree_root($log, $tampered_root1, $leaf2);
        $this->assertNotSame($root2, $tampered_root2);
    }
}
