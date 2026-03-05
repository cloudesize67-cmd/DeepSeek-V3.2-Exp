<?php
/**
 * Unit tests for NoBulkDeleteRule.
 *
 * Run with: vendor/bin/phpunit tests/NoBulkDeleteRuleTest.php
 *
 * No WordPress functions are called, so this test is fully standalone.
 *
 * @package WPG\Tests
 */

declare(strict_types=1);

// Minimal stub for the WPG namespace autoloader (no WP bootstrap needed).
require_once __DIR__ . '/../src/Policy/PolicyRule.php';
require_once __DIR__ . '/../src/Policy/Rules/NoBulkDeleteRule.php';

use PHPUnit\Framework\TestCase;
use WPG\Policy\Rules\NoBulkDeleteRule;

// Stub the WP i18n function used inside the rule.
if (!function_exists('__')) {
    function __(string $text, string $domain = 'default'): string { return $text; }
}

class NoBulkDeleteRuleTest extends TestCase
{
    private NoBulkDeleteRule $rule;

    protected function setUp(): void
    {
        $this->rule = new NoBulkDeleteRule();
    }

    // -----------------------------------------------------------------------
    // name()
    // -----------------------------------------------------------------------

    public function test_name_is_no_bulk_delete(): void
    {
        $this->assertSame('no-bulk-delete', $this->rule->name());
    }

    // -----------------------------------------------------------------------
    // Below limit → ALLOWED
    // -----------------------------------------------------------------------

    public function test_count_below_limit_is_allowed(): void
    {
        $result = $this->rule->evaluate(
            ['action' => 'core/delete-posts', 'count' => 5, 'actor' => []],
            ['limit' => 10]
        );
        $this->assertSame('ALLOWED', $result['verdict']);
    }

    public function test_count_of_one_is_allowed(): void
    {
        $result = $this->rule->evaluate(
            ['count' => 1],
            ['limit' => 10]
        );
        $this->assertSame('ALLOWED', $result['verdict']);
    }

    public function test_count_of_zero_is_allowed(): void
    {
        $result = $this->rule->evaluate(
            ['count' => 0],
            ['limit' => 10]
        );
        $this->assertSame('ALLOWED', $result['verdict']);
    }

    // -----------------------------------------------------------------------
    // At limit → DENIED (>= semantics: "10 or more requires approval")
    // -----------------------------------------------------------------------

    public function test_count_equal_to_limit_is_denied(): void
    {
        $result = $this->rule->evaluate(
            ['count' => 10],
            ['limit' => 10]
        );
        $this->assertSame('DENIED', $result['verdict']);
    }

    public function test_count_above_limit_is_denied(): void
    {
        $result = $this->rule->evaluate(
            ['count' => 847],
            ['limit' => 10]
        );
        $this->assertSame('DENIED', $result['verdict']);
        $this->assertSame(10, $result['limit']);
    }

    // -----------------------------------------------------------------------
    // Custom limit
    // -----------------------------------------------------------------------

    public function test_custom_limit_of_1_denies_count_of_1(): void
    {
        $result = $this->rule->evaluate(
            ['count' => 1],
            ['limit' => 1]
        );
        $this->assertSame('DENIED', $result['verdict']);
    }

    public function test_custom_limit_of_100_allows_99(): void
    {
        $result = $this->rule->evaluate(
            ['count' => 99],
            ['limit' => 100]
        );
        $this->assertSame('ALLOWED', $result['verdict']);
    }

    // -----------------------------------------------------------------------
    // Default limit (no 'limit' key in config)
    // -----------------------------------------------------------------------

    public function test_default_limit_is_10(): void
    {
        $denied = $this->rule->evaluate(['count' => 10], []);
        $this->assertSame('DENIED', $denied['verdict']);
        $this->assertSame(10, $denied['limit']);

        $allowed = $this->rule->evaluate(['count' => 9], []);
        $this->assertSame('ALLOWED', $allowed['verdict']);
    }

    // -----------------------------------------------------------------------
    // Message content
    // -----------------------------------------------------------------------

    public function test_denial_message_contains_counts(): void
    {
        $result = $this->rule->evaluate(['count' => 847], ['limit' => 10]);
        $this->assertStringContainsString('847', $result['message']);
        $this->assertStringContainsString('10',  $result['message']);
    }
}
