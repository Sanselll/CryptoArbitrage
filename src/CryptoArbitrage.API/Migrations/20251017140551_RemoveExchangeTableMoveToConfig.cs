using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Migrations
{
    /// <inheritdoc />
    public partial class RemoveExchangeTableMoveToConfig : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "Executions",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Symbol = table.Column<string>(type: "TEXT", maxLength: 20, nullable: false),
                    Exchange = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    StartedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    StoppedAt = table.Column<DateTime>(type: "TEXT", nullable: true),
                    State = table.Column<int>(type: "INTEGER", nullable: false),
                    FundingEarned = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    PositionSizeUsd = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    SpotOrderId = table.Column<string>(type: "TEXT", maxLength: 100, nullable: false),
                    PerpOrderId = table.Column<string>(type: "TEXT", maxLength: 100, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Executions", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "FundingRates",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Exchange = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    Symbol = table.Column<string>(type: "TEXT", maxLength: 20, nullable: false),
                    Rate = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    AnnualizedRate = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    FundingTime = table.Column<DateTime>(type: "TEXT", nullable: false),
                    NextFundingTime = table.Column<DateTime>(type: "TEXT", nullable: false),
                    RecordedAt = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_FundingRates", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "PerformanceMetrics",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Date = table.Column<DateTime>(type: "TEXT", nullable: false),
                    TotalPnL = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    RealizedPnL = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    UnrealizedPnL = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    TotalFundingFeeReceived = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    TotalFundingFeePaid = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    NetFundingFee = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    TotalTrades = table.Column<int>(type: "INTEGER", nullable: false),
                    WinningTrades = table.Column<int>(type: "INTEGER", nullable: false),
                    LosingTrades = table.Column<int>(type: "INTEGER", nullable: false),
                    WinRate = table.Column<decimal>(type: "TEXT", precision: 5, scale: 2, nullable: false),
                    LargestWin = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    LargestLoss = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    MaxDrawdown = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    AccountBalance = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_PerformanceMetrics", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "Positions",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    ExecutionId = table.Column<int>(type: "INTEGER", nullable: true),
                    Symbol = table.Column<string>(type: "TEXT", maxLength: 20, nullable: false),
                    Exchange = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    Type = table.Column<int>(type: "INTEGER", nullable: false),
                    Side = table.Column<int>(type: "INTEGER", nullable: false),
                    Status = table.Column<int>(type: "INTEGER", nullable: false),
                    EntryPrice = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    ExitPrice = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: true),
                    Quantity = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    Leverage = table.Column<decimal>(type: "TEXT", precision: 5, scale: 2, nullable: false),
                    InitialMargin = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    RealizedPnL = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    UnrealizedPnL = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    TotalFundingFeePaid = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    TotalFundingFeeReceived = table.Column<decimal>(type: "TEXT", precision: 18, scale: 8, nullable: false),
                    OrderId = table.Column<string>(type: "TEXT", maxLength: 100, nullable: true),
                    OpenedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    ClosedAt = table.Column<DateTime>(type: "TEXT", nullable: true),
                    Notes = table.Column<string>(type: "TEXT", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Positions", x => x.Id);
                    table.ForeignKey(
                        name: "FK_Positions_Executions_ExecutionId",
                        column: x => x.ExecutionId,
                        principalTable: "Executions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.SetNull);
                });

            migrationBuilder.CreateIndex(
                name: "IX_Executions_State_StartedAt",
                table: "Executions",
                columns: new[] { "State", "StartedAt" });

            migrationBuilder.CreateIndex(
                name: "IX_Executions_Symbol_Exchange",
                table: "Executions",
                columns: new[] { "Symbol", "Exchange" });

            migrationBuilder.CreateIndex(
                name: "IX_FundingRates_Exchange_Symbol_RecordedAt",
                table: "FundingRates",
                columns: new[] { "Exchange", "Symbol", "RecordedAt" });

            migrationBuilder.CreateIndex(
                name: "IX_PerformanceMetrics_Date",
                table: "PerformanceMetrics",
                column: "Date",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_Positions_ExecutionId_Type",
                table: "Positions",
                columns: new[] { "ExecutionId", "Type" });

            migrationBuilder.CreateIndex(
                name: "IX_Positions_Status_OpenedAt",
                table: "Positions",
                columns: new[] { "Status", "OpenedAt" });

            migrationBuilder.CreateIndex(
                name: "IX_Positions_Symbol_Exchange",
                table: "Positions",
                columns: new[] { "Symbol", "Exchange" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "FundingRates");

            migrationBuilder.DropTable(
                name: "PerformanceMetrics");

            migrationBuilder.DropTable(
                name: "Positions");

            migrationBuilder.DropTable(
                name: "Executions");
        }
    }
}
