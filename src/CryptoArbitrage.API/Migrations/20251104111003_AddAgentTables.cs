using System;
using Microsoft.EntityFrameworkCore.Migrations;
using Npgsql.EntityFrameworkCore.PostgreSQL.Metadata;

#nullable disable

namespace CryptoArbitrage.API.Migrations
{
    /// <inheritdoc />
    public partial class AddAgentTables : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "AgentConfigurations",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    UserId = table.Column<string>(type: "text", nullable: false),
                    MaxLeverage = table.Column<decimal>(type: "numeric(5,2)", precision: 5, scale: 2, nullable: false),
                    TargetUtilization = table.Column<decimal>(type: "numeric(5,4)", precision: 5, scale: 4, nullable: false),
                    MaxPositions = table.Column<int>(type: "integer", nullable: false),
                    PredictionIntervalSeconds = table.Column<int>(type: "integer", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    Notes = table.Column<string>(type: "text", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AgentConfigurations", x => x.Id);
                    table.ForeignKey(
                        name: "FK_AgentConfigurations_AspNetUsers_UserId",
                        column: x => x.UserId,
                        principalTable: "AspNetUsers",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "AgentSessions",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    UserId = table.Column<string>(type: "text", nullable: false),
                    AgentConfigurationId = table.Column<int>(type: "integer", nullable: false),
                    Status = table.Column<int>(type: "integer", nullable: false),
                    StartedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    PausedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    StoppedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    ErrorMessage = table.Column<string>(type: "text", nullable: true),
                    TotalPredictions = table.Column<int>(type: "integer", nullable: false),
                    TotalTrades = table.Column<int>(type: "integer", nullable: false),
                    FinalPnLUsd = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: true),
                    FinalPnLPct = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AgentSessions", x => x.Id);
                    table.ForeignKey(
                        name: "FK_AgentSessions_AgentConfigurations_AgentConfigurationId",
                        column: x => x.AgentConfigurationId,
                        principalTable: "AgentConfigurations",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "FK_AgentSessions_AspNetUsers_UserId",
                        column: x => x.UserId,
                        principalTable: "AspNetUsers",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "AgentStats",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    UserId = table.Column<string>(type: "text", nullable: false),
                    AgentSessionId = table.Column<int>(type: "integer", nullable: true),
                    TotalDecisions = table.Column<int>(type: "integer", nullable: false),
                    HoldDecisions = table.Column<int>(type: "integer", nullable: false),
                    EnterDecisions = table.Column<int>(type: "integer", nullable: false),
                    ExitDecisions = table.Column<int>(type: "integer", nullable: false),
                    TotalTrades = table.Column<int>(type: "integer", nullable: false),
                    WinningTrades = table.Column<int>(type: "integer", nullable: false),
                    LosingTrades = table.Column<int>(type: "integer", nullable: false),
                    WinRate = table.Column<decimal>(type: "numeric(5,2)", precision: 5, scale: 2, nullable: false),
                    TotalPnLUsd = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    TotalPnLPct = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    TodayPnLUsd = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    TodayPnLPct = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    MaxDrawdownPct = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    ActivePositions = table.Column<int>(type: "integer", nullable: false),
                    MaxActivePositions = table.Column<int>(type: "integer", nullable: false),
                    AveragePositionDurationHours = table.Column<decimal>(type: "numeric(18,8)", precision: 18, scale: 8, nullable: false),
                    StatsPeriodStart = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    UpdatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false)
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
                name: "IX_AgentConfigurations_UserId",
                table: "AgentConfigurations",
                column: "UserId");

            migrationBuilder.CreateIndex(
                name: "IX_AgentSessions_AgentConfigurationId",
                table: "AgentSessions",
                column: "AgentConfigurationId");

            migrationBuilder.CreateIndex(
                name: "IX_AgentSessions_StartedAt",
                table: "AgentSessions",
                column: "StartedAt");

            migrationBuilder.CreateIndex(
                name: "IX_AgentSessions_UserId",
                table: "AgentSessions",
                column: "UserId");

            migrationBuilder.CreateIndex(
                name: "IX_AgentSessions_UserId_Status",
                table: "AgentSessions",
                columns: new[] { "UserId", "Status" });

            migrationBuilder.CreateIndex(
                name: "IX_AgentStats_AgentSessionId",
                table: "AgentStats",
                column: "AgentSessionId");

            migrationBuilder.CreateIndex(
                name: "IX_AgentStats_UserId",
                table: "AgentStats",
                column: "UserId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "AgentStats");

            migrationBuilder.DropTable(
                name: "AgentSessions");

            migrationBuilder.DropTable(
                name: "AgentConfigurations");
        }
    }
}
