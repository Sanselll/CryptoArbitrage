using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddFundingIntervalFields : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<decimal>(
                name: "LongFundingIntervalHours",
                table: "Positions",
                type: "numeric",
                nullable: false,
                defaultValue: 8m);

            migrationBuilder.AddColumn<decimal>(
                name: "ShortFundingIntervalHours",
                table: "Positions",
                type: "numeric",
                nullable: false,
                defaultValue: 8m);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "LongFundingIntervalHours",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "ShortFundingIntervalHours",
                table: "Positions");
        }
    }
}
