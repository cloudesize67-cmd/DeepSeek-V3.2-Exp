<?php
/**
 * Contract every policy rule must implement.
 *
 * @package WPG\Policy
 */

declare(strict_types=1);

namespace WPG\Policy;

defined('ABSPATH') || exit;

interface PolicyRule
{
    /** Unique machine-readable identifier (matches the 'rule' field in config). */
    public function name(): string;

    /**
     * Evaluate the action context against this rule.
     *
     * @param  array<string, mixed> $context    Normalised ActionContext.
     * @param  array<string, mixed> $policy_cfg The raw policy configuration row.
     * @return array{verdict: string, message: string, limit?: int}
     *         verdict is 'ALLOWED' or 'DENIED'.
     */
    public function evaluate(array $context, array $policy_cfg): array;
}
