using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Migrations
{
    /// <inheritdoc />
    public partial class AddExecutionMetadataAndRequireExecutionId : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // Step 1: Add new columns to Executions table first
            migrationBuilder.AddColumn<string>(
                name: "LongExchange",
                table: "Executions",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "ShortExchange",
                table: "Executions",
                type: "text",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "Strategy",
                table: "Executions",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "SubType",
                table: "Executions",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            // Step 2: Create orphaned executions for positions without ExecutionId (data migration)
            migrationBuilder.Sql(@"
                DO $$
                DECLARE
                    orphan_pos RECORD;
                    new_execution_id INTEGER;
                BEGIN
                    -- For each position without an ExecutionId, create an orphaned execution
                    FOR orphan_pos IN
                        SELECT ""Id"", ""UserId"", ""Symbol"", ""Exchange"", ""OpenedAt"", ""ClosedAt"",
                               ""Status"", ""InitialMargin""
                        FROM ""Positions""
                        WHERE ""ExecutionId"" IS NULL
                    LOOP
                        -- Create orphaned execution
                        INSERT INTO ""Executions"" (
                            ""UserId"", ""Symbol"", ""Exchange"", ""StartedAt"", ""StoppedAt"",
                            ""State"", ""PositionSizeUsd"", ""FundingEarned"",
                            ""Strategy"", ""SubType""
                        ) VALUES (
                            orphan_pos.""UserId"",
                            orphan_pos.""Symbol"",
                            orphan_pos.""Exchange"",
                            orphan_pos.""OpenedAt"",
                            orphan_pos.""ClosedAt"",
                            CASE WHEN orphan_pos.""Status"" = 1 THEN 1 ELSE 0 END, -- Closed=1, Running=0
                            orphan_pos.""InitialMargin"",
                            0, -- FundingEarned default
                            0, -- Strategy = CrossExchange (default)
                            1  -- SubType = CrossExchangeFuturesFutures (default)
                        ) RETURNING ""Id"" INTO new_execution_id;

                        -- Link position to the new execution
                        UPDATE ""Positions""
                        SET ""ExecutionId"" = new_execution_id
                        WHERE ""Id"" = orphan_pos.""Id"";

                        RAISE NOTICE 'Created orphaned execution % for position %', new_execution_id, orphan_pos.""Id"";
                    END LOOP;
                END $$;
            ");

            // Step 3: Now make ExecutionId NOT NULL (safe after data migration)
            migrationBuilder.AlterColumn<int>(
                name: "ExecutionId",
                table: "Positions",
                type: "integer",
                nullable: false,
                defaultValue: 0,
                oldClrType: typeof(int),
                oldType: "integer",
                oldNullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "LongExchange",
                table: "Executions");

            migrationBuilder.DropColumn(
                name: "ShortExchange",
                table: "Executions");

            migrationBuilder.DropColumn(
                name: "Strategy",
                table: "Executions");

            migrationBuilder.DropColumn(
                name: "SubType",
                table: "Executions");

            migrationBuilder.AlterColumn<int>(
                name: "ExecutionId",
                table: "Positions",
                type: "integer",
                nullable: true,
                oldClrType: typeof(int),
                oldType: "integer");
        }
    }
}
