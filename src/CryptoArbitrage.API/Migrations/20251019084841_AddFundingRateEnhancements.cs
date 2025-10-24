using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Migrations
{
    /// <inheritdoc />
    public partial class AddFundingRateEnhancements : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "Direction",
                table: "FundingRates",
                type: "integer",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "FundingCap",
                table: "FundingRates",
                type: "numeric",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "FundingFloor",
                table: "FundingRates",
                type: "numeric",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "PreviousAnnualizedRate",
                table: "FundingRates",
                type: "numeric",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "PreviousRate",
                table: "FundingRates",
                type: "numeric",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Direction",
                table: "FundingRates");

            migrationBuilder.DropColumn(
                name: "FundingCap",
                table: "FundingRates");

            migrationBuilder.DropColumn(
                name: "FundingFloor",
                table: "FundingRates");

            migrationBuilder.DropColumn(
                name: "PreviousAnnualizedRate",
                table: "FundingRates");

            migrationBuilder.DropColumn(
                name: "PreviousRate",
                table: "FundingRates");
        }
    }
}
