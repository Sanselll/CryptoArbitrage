using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Data.Migrations
{
    /// <inheritdoc />
    public partial class UpdatePredictionIntervalTo1Hour : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // Update all existing AgentConfigurations to use 1-hour prediction interval
            migrationBuilder.Sql(@"
                UPDATE ""AgentConfigurations""
                SET ""PredictionIntervalSeconds"" = 3600,
                    ""UpdatedAt"" = NOW()
                WHERE ""PredictionIntervalSeconds"" < 3600;
            ");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {

        }
    }
}
