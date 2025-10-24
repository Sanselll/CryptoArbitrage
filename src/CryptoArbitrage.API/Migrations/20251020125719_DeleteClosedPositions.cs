using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Migrations
{
    /// <inheritdoc />
    public partial class DeleteClosedPositions : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // Clean up all existing positions with Status = 1 (Closed)
            // Going forward, we delete positions instead of marking them as closed
            migrationBuilder.Sql("DELETE FROM \"Positions\" WHERE \"Status\" = 1;");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {

        }
    }
}
