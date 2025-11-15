using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddLastKnownAprToPosition : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<decimal>(
                name: "LastKnownApr",
                table: "Positions",
                type: "numeric",
                nullable: false,
                defaultValue: 0m);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "LastKnownApr",
                table: "Positions");
        }
    }
}
