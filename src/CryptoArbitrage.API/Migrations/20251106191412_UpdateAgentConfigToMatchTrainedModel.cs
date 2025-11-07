using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Data.Migrations
{
    /// <inheritdoc />
    public partial class UpdateAgentConfigToMatchTrainedModel : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // Update existing AgentConfiguration records to match trained model parameters
            // Training used: max_leverage=2.0, target_utilization=0.8, max_positions=2, step_minutes=5 (300 seconds)
            // Only update if table exists (handles fresh database installations)
            migrationBuilder.Sql(@"
                DO $$
                BEGIN
                    IF EXISTS (SELECT FROM information_schema.tables
                              WHERE table_schema = 'public'
                              AND table_name = 'AgentConfiguration') THEN
                        UPDATE ""AgentConfiguration""
                        SET
                            ""MaxLeverage"" = 2.0,
                            ""TargetUtilization"" = 0.8,
                            ""MaxPositions"" = 2,
                            ""PredictionIntervalSeconds"" = 300,
                            ""UpdatedAt"" = NOW()
                        WHERE ""MaxLeverage"" != 2.0
                           OR ""TargetUtilization"" != 0.8
                           OR ""MaxPositions"" != 2
                           OR ""PredictionIntervalSeconds"" != 300;
                    END IF;
                END $$;
            ");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            // Revert to previous default values (only if table exists)
            migrationBuilder.Sql(@"
                DO $$
                BEGIN
                    IF EXISTS (SELECT FROM information_schema.tables
                              WHERE table_schema = 'public'
                              AND table_name = 'AgentConfiguration') THEN
                        UPDATE ""AgentConfiguration""
                        SET
                            ""MaxLeverage"" = 1.0,
                            ""TargetUtilization"" = 0.9,
                            ""MaxPositions"" = 3,
                            ""PredictionIntervalSeconds"" = 3600,
                            ""UpdatedAt"" = NOW();
                    END IF;
                END $$;
            ");
        }
    }
}
