using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ExchangeController : ControllerBase
{
    private readonly ArbitrageDbContext _context;
    private readonly ILogger<ExchangeController> _logger;

    public ExchangeController(ArbitrageDbContext context, ILogger<ExchangeController> logger)
    {
        _context = context;
        _logger = logger;
    }

    [HttpGet]
    public async Task<ActionResult<List<Exchange>>> GetExchanges()
    {
        return await _context.Exchanges.ToListAsync();
    }

    [HttpGet("{id}")]
    public async Task<ActionResult<Exchange>> GetExchange(int id)
    {
        var exchange = await _context.Exchanges.FindAsync(id);
        if (exchange == null)
            return NotFound();

        return exchange;
    }

    [HttpPut("{id}")]
    public async Task<IActionResult> UpdateExchange(int id, Exchange exchange)
    {
        if (id != exchange.Id)
            return BadRequest();

        exchange.UpdatedAt = DateTime.UtcNow;
        _context.Entry(exchange).State = EntityState.Modified;

        try
        {
            await _context.SaveChangesAsync();
        }
        catch (DbUpdateConcurrencyException)
        {
            if (!await _context.Exchanges.AnyAsync(e => e.Id == id))
                return NotFound();
            throw;
        }

        return NoContent();
    }

    [HttpPost("{id}/toggle")]
    public async Task<IActionResult> ToggleExchange(int id)
    {
        var exchange = await _context.Exchanges.FindAsync(id);
        if (exchange == null)
            return NotFound();

        exchange.IsEnabled = !exchange.IsEnabled;
        exchange.UpdatedAt = DateTime.UtcNow;
        await _context.SaveChangesAsync();

        return Ok(new { enabled = exchange.IsEnabled });
    }
}
