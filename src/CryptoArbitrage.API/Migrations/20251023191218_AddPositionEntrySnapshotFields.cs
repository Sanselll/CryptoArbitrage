using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Migrations
{
    /// <inheritdoc />
    public partial class AddPositionEntrySnapshotFields : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<decimal>(
                name: "EntryConfidenceScore",
                table: "Positions",
                type: "numeric",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "EntryFundingRate",
                table: "Positions",
                type: "numeric",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "EntryPerpPrice",
                table: "Positions",
                type: "numeric",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "EntrySpotPrice",
                table: "Positions",
                type: "numeric",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "EntrySpread",
                table: "Positions",
                type: "numeric",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "MaxHoldingHours",
                table: "Positions",
                type: "numeric",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "ProfitTargetPercent",
                table: "Positions",
                type: "numeric",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "RecommendedStrategy",
                table: "Positions",
                type: "text",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "EntryConfidenceScore",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "EntryFundingRate",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "EntryPerpPrice",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "EntrySpotPrice",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "EntrySpread",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "MaxHoldingHours",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "ProfitTargetPercent",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "RecommendedStrategy",
                table: "Positions");
        }
    }
}
