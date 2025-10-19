using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Migrations
{
    /// <inheritdoc />
    public partial class AddFundingIntervalAndAverage : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<decimal>(
                name: "Average3DayRate",
                table: "FundingRates",
                type: "numeric",
                nullable: true);

            migrationBuilder.AddColumn<int>(
                name: "FundingIntervalHours",
                table: "FundingRates",
                type: "integer",
                nullable: false,
                defaultValue: 0);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Average3DayRate",
                table: "FundingRates");

            migrationBuilder.DropColumn(
                name: "FundingIntervalHours",
                table: "FundingRates");
        }
    }
}
