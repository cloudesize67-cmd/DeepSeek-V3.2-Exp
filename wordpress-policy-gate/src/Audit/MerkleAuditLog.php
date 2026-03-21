<?php
/**
 * Merkle-tree authenticated audit log.
 *
 * Every audit record is a leaf in an append-only Merkle tree.  Each leaf hash
 * is computed as:
 *
 *   leaf_hash = SHA-256( prev_root || action_id || timestamp || payload_json )
 *
 * The running tree root is recomputed after every insertion so the entire
 * history can be verified by replaying the leaf sequence.
 *
 * Verification:
 *   Given rows ordered by id, replay:
 *     root_0 = sha256('' || row1.leaf_hash)
 *     root_n = sha256(root_{n-1} || row_n.leaf_hash)
 *   The final root must equal the tree_root stored in the last row.
 *
 * @package WPG\Audit
 */

declare(strict_types=1);

namespace WPG\Audit;

defined('ABSPATH') || exit;

final class MerkleAuditLog
{
    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * Append one record to the audit log.
     *
     * @param  array<string, mixed> $entry  Must contain keys documented below.
     * @return int  Auto-increment ID of the newly inserted row.
     *
     * Required $entry keys:
     *   action_id   (string)   UUID for this specific action invocation
     *   action_name (string)   e.g. 'core/delete-posts'
     *   actor_id    (int)      WP user ID
     *   actor_role  (string)   comma-separated roles
     *   verdict     (string)   'ALLOWED' | 'DENIED'
     *   policy_name (string)   which policy triggered
     *   item_count  (int)      number of items involved
     *   context     (array)    full ActionContext for forensics
     */
    public function record(array $entry): int
    {
        global $wpdb;

        $table = $wpdb->prefix . \WPG\Installer::TABLE;

        // Fetch the most-recent tree root (or empty string for the genesis leaf).
        $prev_root = (string) $wpdb->get_var("SELECT tree_root FROM {$table} ORDER BY id DESC LIMIT 1");

        $timestamp    = current_time('mysql', true); // UTC
        $context_json = wp_json_encode($entry['context'] ?? [], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
        $leaf_hash    = $this->compute_leaf_hash($prev_root, $entry['action_id'], $timestamp, (string) $context_json);
        $tree_root    = $this->compute_tree_root($prev_root, $leaf_hash);

        $wpdb->insert(
            $table,
            [
                'created_at'   => $timestamp,
                'action_id'    => $entry['action_id'],
                'action_name'  => $entry['action_name'],
                'actor_id'     => (int) $entry['actor_id'],
                'actor_role'   => $entry['actor_role'],
                'verdict'      => $entry['verdict'],
                'policy_name'  => $entry['policy_name'],
                'item_count'   => (int) $entry['item_count'],
                'context_json' => $context_json,
                'leaf_hash'    => $leaf_hash,
                'tree_root'    => $tree_root,
            ],
            ['%s', '%s', '%s', '%d', '%s', '%s', '%s', '%d', '%s', '%s', '%s']
        );

        return (int) $wpdb->insert_id;
    }

    // -----------------------------------------------------------------------
    // Verification
    // -----------------------------------------------------------------------

    /**
     * Verify the integrity of the entire audit log.
     *
     * Replays all leaf hashes in insertion order and checks that every stored
     * tree_root matches the recomputed running root.
     *
     * @return array{valid: bool, checked: int, first_tampered_id: int|null}
     */
    public function verify(): array
    {
        global $wpdb;

        $table = $wpdb->prefix . \WPG\Installer::TABLE;
        $rows  = $wpdb->get_results(
            "SELECT id, leaf_hash, tree_root FROM {$table} ORDER BY id ASC",
            ARRAY_A
        );

        $running_root      = '';
        $first_tampered_id = null;

        foreach ($rows as $row) {
            $running_root = $this->compute_tree_root($running_root, $row['leaf_hash']);
            if (!hash_equals($running_root, $row['tree_root'])) {
                $first_tampered_id = (int) $row['id'];
                break;
            }
        }

        return [
            'valid'              => $first_tampered_id === null,
            'checked'            => count($rows),
            'first_tampered_id'  => $first_tampered_id,
        ];
    }

    /**
     * Return recent audit records (newest first).
     *
     * @param  int  $limit  Max records to return.
     * @return array<int, array<string, mixed>>
     */
    public function recent(int $limit = 50): array
    {
        global $wpdb;
        $table = $wpdb->prefix . \WPG\Installer::TABLE;
        $limit = max(1, min($limit, 500));

        return (array) $wpdb->get_results(
            $wpdb->prepare(
                "SELECT * FROM {$table} ORDER BY id DESC LIMIT %d",
                $limit
            ),
            ARRAY_A
        );
    }

    // -----------------------------------------------------------------------
    // Cryptographic helpers
    // -----------------------------------------------------------------------

    /**
     * Compute the SHA-256 leaf hash for a single audit entry.
     *
     * The pre-image binds the previous tree root so that any tampering with
     * earlier records invalidates all subsequent hashes.
     */
    private function compute_leaf_hash(
        string $prev_root,
        string $action_id,
        string $timestamp,
        string $payload_json
    ): string {
        return hash('sha256', $prev_root . "\x00" . $action_id . "\x00" . $timestamp . "\x00" . $payload_json);
    }

    /**
     * Combine the previous running root with a new leaf to get the next root.
     *
     * Uses the standard "parent = H(left || right)" construction.
     */
    private function compute_tree_root(string $prev_root, string $leaf_hash): string
    {
        return hash('sha256', $prev_root . $leaf_hash);
    }
}
