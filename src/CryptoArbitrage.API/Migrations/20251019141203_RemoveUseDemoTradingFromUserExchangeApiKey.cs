using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CryptoArbitrage.API.Migrations
{
    /// <inheritdoc />
    public partial class RemoveUseDemoTradingFromUserExchangeApiKey : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "UseDemoTrading",
                table: "UserExchangeApiKeys");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "UseDemoTrading",
                table: "UserExchangeApiKeys",
                type: "boolean",
                nullable: false,
                defaultValue: false);
        }
    }
}
