using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Migrations
{
    /// <inheritdoc />
    public partial class AddUniqueFundingRateIndex : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_FundingRates_Exchange_Symbol_RecordedAt",
                table: "FundingRates");

            // Clear all existing funding rate data before creating unique index
            // This data is transient market data that gets refreshed constantly anyway
            migrationBuilder.Sql(@"DELETE FROM ""FundingRates"";");

            migrationBuilder.CreateIndex(
                name: "IX_FundingRates_Exchange_Symbol",
                table: "FundingRates",
                columns: new[] { "Exchange", "Symbol" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_FundingRates_RecordedAt",
                table: "FundingRates",
                column: "RecordedAt");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_FundingRates_Exchange_Symbol",
                table: "FundingRates");

            migrationBuilder.DropIndex(
                name: "IX_FundingRates_RecordedAt",
                table: "FundingRates");

            migrationBuilder.CreateIndex(
                name: "IX_FundingRates_Exchange_Symbol_RecordedAt",
                table: "FundingRates",
                columns: new[] { "Exchange", "Symbol", "RecordedAt" });
        }
    }
}
