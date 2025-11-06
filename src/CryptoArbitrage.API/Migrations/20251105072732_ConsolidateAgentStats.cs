using System;
using Microsoft.EntityFrameworkCore.Migrations;
using Npgsql.EntityFrameworkCore.PostgreSQL.Metadata;

#nullable disable

namespace CryptoArbitrage.API.Data.Migrations
{
    /// <inheritdoc />
    public partial class ConsolidateAgentStats : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "AgentStats");

            migrationBuilder.DropColumn(
                name: "FinalPnLPct",
                table: "AgentSessions");

            migrationBuilder.DropColumn(
                name: "FinalPnLUsd",
                table: "AgentSessions");

            migrationBuilder.RenameColumn(
                name: "RealizedPnL",
                table: "Positions",
                newName: "TradingFeesUsd");

            migrationBuilder.RenameColumn(
                name: "TotalTrades",
                table: "AgentSessions",
                newName: "WinningTrades");

            migrationBuilder.RenameColumn(
                name: "TotalPredictions",
                table: "AgentSessions",
                newName: "MaxActivePositions");

            migrationBuilder.AddColumn<Guid>(
                name: "AgentSessionId",
                table: "Positions",
                type: "uuid",
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "FundingEarnedUsd",
                table: "Positions",
                type: "numeric(18,8)",
                precision: 18,
                scale: 8,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<decimal>(
                name: "PricePnLUsd",
                table: "Positions",
                type: "numeric(18,8)",
                precision: 18,
                scale: 8,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<decimal>(
                name: "RealizedPnLPct",
                table: "Positions",
                type: "numeric(18,8)",
                precision: 18,
                scale: 8,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<decimal>(
                name: "RealizedPnLUsd",
                table: "Positions",
                type: "numeric(18,8)",
                precision: 18,
                scale: 8,
                nullable: false,
                defaultValue: 0m);

            // Clear existing sessions since we're changing PK type and consolidating stats
            migrationBuilder.Sql("TRUNCATE TABLE \"AgentSessions\" CASCADE;");

            // Manually drop identity constraint first, then change type to UUID
            migrationBuilder.Sql(@"
                ALTER TABLE ""AgentSessions"" ALTER COLUMN ""Id"" DROP IDENTITY IF EXISTS;
                ALTER TABLE ""AgentSessions"" ALTER COLUMN ""Id"" TYPE uuid USING gen_random_uuid();
            ");

            migrationBuilder.AddColumn<int>(
                name: "ActivePositions",
                table: "AgentSessions",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "EnterDecisions",
                table: "AgentSessions",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "ExitDecisions",
                table: "AgentSessions",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "HoldDecisions",
                table: "AgentSessions",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "LosingTrades",
                table: "AgentSessions",
                type: "integer",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<decimal>(
                name: "SessionPnLPct",
                table: "AgentSessions",
                type: "numeric(18,8)",
                precision: 18,
                scale: 8,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<decimal>(
                name: "SessionPnLUsd",
                table: "AgentSessions",
                type: "numeric(18,8)",
                precision: 18,
                scale: 8,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.CreateIndex(
                name: "IX_Positions_AgentSessionId",
                table: "Positions",
                column: "AgentSessionId");

            migrationBuilder.AddForeignKey(
                name: "FK_Positions_AgentSessions_AgentSessionId",
                table: "Positions",
                column: "AgentSessionId",
                principalTable: "AgentSessions",
                principalColumn: "Id",
                onDelete: ReferentialAction.SetNull);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Positions_AgentSessions_AgentSessionId",
                table: "Positions");

            migrationBuilder.DropIndex(
                name: "IX_Positions_AgentSessionId",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "AgentSessionId",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "FundingEarnedUsd",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "PricePnLUsd",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "RealizedPnLPct",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "RealizedPnLUsd",
                table: "Positions");

            migrationBuilder.DropColumn(
                name: "ActivePositions",
                table: "AgentSessions");

            migrationBuilder.DropColumn(
                name: "EnterDecisions",
                table: "AgentSessions");

            migrationBuilder.DropColumn(
                name: "ExitDecisions",
                table: "AgentSessions");

            migrationBuilder.DropColumn(
                name: "HoldDecisions",
                table: "AgentSessions");

            migrationBuilder.DropColumn(
                name: "LosingTrades",
                table: "AgentSessions");

            migrationBuilder.DropColumn(
                name: "SessionPnLPct",
                table: "AgentSessions");

            migrationBuilder.DropColumn(
                name: "SessionPnLUsd",
                table: "AgentSessions");

            migrationBuilder.RenameColumn(
                name: "TradingFeesUsd",
                table: "Positions",
                newName: "RealizedPnL");

            migrationBuilder.RenameColumn(
                name: "WinningTrades",
                table: "AgentSessions",
                newName: "TotalTrades");

            migrationBuilder.RenameColumn(
                name: "MaxActivePositions",
                table: "AgentSessions",
                newName: "TotalPredictions");

            // Clear sessions before reverting PK type
            migrationBuilder.Sql("TRUNCATE TABLE \"AgentSessions\" CASCADE;");

            // Manually change type back to int, then add identity constraint
            migrationBuilder.Sql(@"
                ALTER TABLE ""AgentSessions"" ALTER COLUMN ""Id"" TYPE integer USING 0;
                ALTER TABLE ""AgentSessions"" ALTER COLUMN ""Id"" ADD GENERATED BY DEFAULT AS IDENTITY;
            ");

            migrationBuilder.AddColumn<decimal>(
                name: "FinalPnLPct",
                table: "AgentSessions",
                type: "numeric(18,8)",
                precision: 18,
                scale: 8,
                nullable: true);

            migrationBuilder.AddColumn<decimal>(
                name: "FinalPnLUsd",
                table: "AgentSessions",
                type: "numeric(18,8)",
                precision: 18,
                scale: 8,
                nullable: true);

            migrationBuilder.CreateTable(
                name: "AgentStats",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    AgentSessionId = table.Column<int>(type: "integer", nullable: true),
                    UserId = table.Column<string>(type: "text", nullable: false),
                    ActivePositions = table.Column<int>(type: "integer", nullable: false),
                    AveragePositionDurationHours = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    EnterDecisions = table.Column<int>(type: "integer", nullable: false),
                    ExitDecisions = table.Column<int>(type: "integer", nullable: false),
                    HoldDecisions = table.Column<int>(type: "integer", nullable: false),
                    LosingTrades = table.Column<int>(type: "integer", nullable: false),
                    MaxActivePositions = table.Column<int>(type: "integer", nullable: false),
                    MaxDrawdownPct = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    StatsPeriodStart = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    TodayPnLPct = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    TodayPnLUsd = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    TotalDecisions = table.Column<int>(type: "integer", nullable: false),
                    TotalPnLPct = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    TotalPnLUsd = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    TotalTrades = table.Column<int>(type: "integer", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    WinRate = table.Column<decimal>(type: "numeric(5,2)", precision: 5, scale: 2, nullable: false),
                    WinningTrades = table.Column<int>(type: "integer", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AgentStats", x => x.Id);
                    table.ForeignKey(
                        name: "FK_AgentStats_AgentSessions_AgentSessionId",
                        column: x => x.AgentSessionId,
                        principalTable: "AgentSessions",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.SetNull);
                    table.ForeignKey(
                        name: "FK_AgentStats_AspNetUsers_UserId",
                        column: x => x.UserId,
                        principalTable: "AspNetUsers",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_AgentStats_AgentSessionId",
                table: "AgentStats",
                column: "AgentSessionId");

            migrationBuilder.CreateIndex(
                name: "IX_AgentStats_UserId",
                table: "AgentStats",
                column: "UserId");
        }
    }
}
