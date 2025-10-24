using System;
using Microsoft.EntityFrameworkCore.Migrations;
using Npgsql.EntityFrameworkCore.PostgreSQL.Metadata;

#nullable disable

namespace CryptoArbitrage.API.Migrations
{
    /// <inheritdoc />
    public partial class RemoveFundingRatesTable : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "FundingRates");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "FundingRates",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    AnnualizedRate = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    Average3DayRate = table.Column<decimal>(type: "numeric", nullable: true),
                    Direction = table.Column<int>(type: "integer", nullable: true),
                    Exchange = table.Column<string>(type: "character varying(50)", maxLength: 50, nullable: false),
                    FundingCap = table.Column<decimal>(type: "numeric", nullable: true),
                    FundingFloor = table.Column<decimal>(type: "numeric", nullable: true),
                    FundingIntervalHours = table.Column<int>(type: "integer", nullable: false),
                    FundingTime = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    NextFundingTime = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    PreviousAnnualizedRate = table.Column<decimal>(type: "numeric", nullable: true),
                    PreviousRate = table.Column<decimal>(type: "numeric", nullable: true),
                    Rate = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    RecordedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    Symbol = table.Column<string>(type: "character varying(20)", maxLength: 20, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_FundingRates", x => x.Id);
                });

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
    }
}
