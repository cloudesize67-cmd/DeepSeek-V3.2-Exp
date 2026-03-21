<?php
/**
 * Rule: no-bulk-delete
 *
 * Denies any delete/trash action that targets more items than the configured
 * limit (default: 10). The limit can be overridden per-policy in the DB option.
 *
 * @package WPG\Policy\Rules
 */

declare(strict_types=1);

namespace WPG\Policy\Rules;

use WPG\Policy\PolicyRule;

defined('ABSPATH') || exit;

final class NoBulkDeleteRule implements PolicyRule
{
    public function name(): string
    {
        return 'no-bulk-delete';
    }

    /**
     * {@inheritDoc}
     */
    public function evaluate(array $context, array $policy_cfg): array
    {
        $limit = (int) ($policy_cfg['limit'] ?? 10);
        $count = (int) ($context['count'] ?? 0);

        if ($count >= $limit) {
            return [
                'verdict' => 'DENIED',
                'limit'   => $limit,
                'message' => sprintf(
                    /* translators: 1: item count, 2: policy limit */
                    __(
                        'Bulk deletion of %1$d items exceeds the policy limit of %2$d. '
                        . 'Human approval is required for bulk deletions of %2$d or more items.',
                        'wpg'
                    ),
                    $count,
                    $limit
                ),
            ];
        }

        return ['verdict' => 'ALLOWED', 'message' => 'Within bulk-delete limit.'];
    }
}
