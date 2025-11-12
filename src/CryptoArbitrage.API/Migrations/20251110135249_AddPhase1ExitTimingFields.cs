using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddPhase1ExitTimingFields : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<decimal>(
                name: "EntryApr",
                table: "Positions",
                type: "numeric(18,8)",
                precision: 18,
                scale: 8,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<DateTime>(
                name: "LastPnlSnapshotTime",
                table: "Positions",
                type: "timestamp with time zone",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "PeakPnlPct",
                table: "Positions",
                type: "numeric(18,8)",
                precision: 18,
                scale: 8,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<string>(
                name: "PnlHistoryJson",
                table: "Positions",
                type: "jsonb",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "EntryApr",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "LastPnlSnapshotTime",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "PeakPnlPct",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "PnlHistoryJson",
                table: "Positions");
        }
    }
}
