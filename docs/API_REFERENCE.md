# API Reference Documentation

## Base URL

**Development**: `http://localhost:5000`
**Production**: `https://your-domain.com`

## Authentication

Currently, the API does not require authentication. For production deployments, implement JWT-based authentication.

**Future Header**:
```
Authorization: Bearer <your-jwt-token>
```

## Table of Contents

1. [Exchange Endpoints](#exchange-endpoints)
2. [Position Endpoints](#position-endpoints)
3. [Opportunity Endpoints](#opportunity-endpoints)
4. [SignalR Hub](#signalr-hub)
5. [Error Responses](#error-responses)
6. [Rate Limiting](#rate-limiting)

---

## Exchange Endpoints

### List All Exchanges

Retrieve all configured exchanges.

**Endpoint**: `GET /api/exchange`

**Response**: `200 OK`
```json
[
  {
    "id": 1,
    "name": "Binance",
    "apiKey": "your-api-key",
    "apiSecret": "your-api-secret",
    "isEnabled": true,
    "createdAt": "2025-01-15T10:00:00Z",
    "updatedAt": "2025-01-15T10:00:00Z"
  },
  {
    "id": 2,
    "name": "Bybit",
    "apiKey": "your-api-key",
    "apiSecret": "your-api-secret",
    "isEnabled": false,
    "createdAt": "2025-01-15T10:00:00Z",
    "updatedAt": "2025-01-15T10:00:00Z"
  }
]
```

**Example Request**:
```bash
curl -X GET "http://localhost:5000/api/exchange"
```

---

### Get Exchange by ID

Retrieve a specific exchange configuration.

**Endpoint**: `GET /api/exchange/{id}`

**Path Parameters**:
- `id` (integer, required): Exchange ID

**Response**: `200 OK`
```json
{
  "id": 1,
  "name": "Binance",
  "apiKey": "your-api-key",
  "apiSecret": "your-api-secret",
  "isEnabled": true,
  "createdAt": "2025-01-15T10:00:00Z",
  "updatedAt": "2025-01-15T10:00:00Z"
}
```

**Error Response**: `404 Not Found`
```json
{
  "error": "Exchange not found"
}
```

**Example Request**:
```bash
curl -X GET "http://localhost:5000/api/exchange/1"
```

---

### Update Exchange

Update exchange configuration and credentials.

**Endpoint**: `PUT /api/exchange/{id}`

**Path Parameters**:
- `id` (integer, required): Exchange ID

**Request Body**:
```json
{
  "id": 1,
  "name": "Binance",
  "apiKey": "new-api-key",
  "apiSecret": "new-api-secret",
  "isEnabled": true
}
```

**Response**: `204 No Content`

**Error Responses**:
- `400 Bad Request`: ID mismatch or validation error
- `404 Not Found`: Exchange doesn't exist

**Example Request**:
```bash
curl -X PUT "http://localhost:5000/api/exchange/1" \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "name": "Binance",
    "apiKey": "new-key",
    "apiSecret": "new-secret",
    "isEnabled": true
  }'
```

---

### Toggle Exchange

Enable or disable an exchange.

**Endpoint**: `POST /api/exchange/{id}/toggle`

**Path Parameters**:
- `id` (integer, required): Exchange ID

**Response**: `200 OK`
```json
{
  "enabled": true
}
```

**Error Response**: `404 Not Found`

**Example Request**:
```bash
curl -X POST "http://localhost:5000/api/exchange/1/toggle"
```

---

## Position Endpoints

### List All Positions

Retrieve all positions with optional status filtering.

**Endpoint**: `GET /api/position`

**Query Parameters**:
- `status` (string, optional): Filter by status (`Open`, `Closed`, `Liquidated`, `PartiallyFilled`)

**Response**: `200 OK`
```json
[
  {
    "id": 1,
    "exchange": "Binance",
    "symbol": "BTCUSDT",
    "side": 0,
    "status": 0,
    "entryPrice": 43250.50,
    "exitPrice": null,
    "quantity": 0.1,
    "leverage": 5.0,
    "initialMargin": 865.01,
    "realizedPnL": 0.0,
    "unrealizedPnL": 125.50,
    "totalFundingFeePaid": 2.15,
    "totalFundingFeeReceived": 8.30,
    "netFundingFee": 6.15,
    "openedAt": "2025-01-15T12:30:00Z",
    "closedAt": null
  }
]
```

**Side Enum**:
- `0`: Long
- `1`: Short

**Status Enum**:
- `0`: Open
- `1`: Closed
- `2`: Liquidated
- `3`: PartiallyFilled

**Example Requests**:
```bash
# All positions
curl -X GET "http://localhost:5000/api/position"

# Only open positions
curl -X GET "http://localhost:5000/api/position?status=Open"
```

---

### Get Position by ID

Retrieve detailed information about a specific position.

**Endpoint**: `GET /api/position/{id}`

**Path Parameters**:
- `id` (integer, required): Position ID

**Response**: `200 OK`
```json
{
  "id": 1,
  "exchange": "Binance",
  "symbol": "BTCUSDT",
  "side": 0,
  "status": 0,
  "entryPrice": 43250.50,
  "exitPrice": null,
  "quantity": 0.1,
  "leverage": 5.0,
  "initialMargin": 865.01,
  "realizedPnL": 0.0,
  "unrealizedPnL": 125.50,
  "totalFundingFeePaid": 2.15,
  "totalFundingFeeReceived": 8.30,
  "netFundingFee": 6.15,
  "openedAt": "2025-01-15T12:30:00Z",
  "closedAt": null
}
```

**Error Response**: `404 Not Found`

**Example Request**:
```bash
curl -X GET "http://localhost:5000/api/position/1"
```

---

## Opportunity Endpoints

### List Arbitrage Opportunities

Retrieve arbitrage opportunities with optional filtering.

**Endpoint**: `GET /api/opportunity`

**Query Parameters**:
- `limit` (integer, optional, default: 50): Maximum results
- `status` (string, optional): Filter by status (`Detected`, `Executed`, `Expired`, `Ignored`)

**Response**: `200 OK`
```json
[
  {
    "id": 1,
    "symbol": "BTCUSDT",
    "longExchange": "Binance",
    "shortExchange": "Bybit",
    "longFundingRate": -0.0001,
    "shortFundingRate": 0.0005,
    "spreadRate": 0.0006,
    "annualizedSpread": 0.6570,
    "estimatedProfitPercentage": 65.70,
    "status": 0,
    "detectedAt": "2025-01-15T14:30:00Z",
    "executedAt": null
  }
]
```

**Status Enum**:
- `0`: Detected
- `1`: Executed
- `2`: Expired
- `3`: Ignored

**Example Requests**:
```bash
# Get all opportunities
curl -X GET "http://localhost:5000/api/opportunity"

# Get only detected opportunities
curl -X GET "http://localhost:5000/api/opportunity?status=Detected"

# Get top 10 opportunities
curl -X GET "http://localhost:5000/api/opportunity?limit=10"
```

---

### Get Active Opportunities

Retrieve currently active arbitrage opportunities (detected within last 10 minutes).

**Endpoint**: `GET /api/opportunity/active`

**Response**: `200 OK`
```json
[
  {
    "id": 5,
    "symbol": "ETHUSDT",
    "longExchange": "Bybit",
    "shortExchange": "Binance",
    "longFundingRate": 0.0002,
    "shortFundingRate": 0.0008,
    "spreadRate": 0.0006,
    "annualizedSpread": 0.6570,
    "estimatedProfitPercentage": 65.70,
    "status": 0,
    "detectedAt": "2025-01-15T14:45:00Z",
    "executedAt": null
  }
]
```

**Sorting**: Results sorted by `annualizedSpread` (descending)

**Example Request**:
```bash
curl -X GET "http://localhost:5000/api/opportunity/active"
```

---

## SignalR Hub

### Connection

**Hub URL**: `http://localhost:5000/arbitragehub`

**Connection Example (JavaScript/TypeScript)**:
```typescript
import * as signalR from '@microsoft/signalr';

const connection = new signalR.HubConnectionBuilder()
  .withUrl('http://localhost:5000/arbitragehub')
  .withAutomaticReconnect()
  .configureLogging(signalR.LogLevel.Information)
  .build();

await connection.start();
```

---

### Server-to-Client Events

#### ReceiveFundingRates

Real-time funding rate updates from all exchanges.

**Event Name**: `ReceiveFundingRates`

**Payload**: `FundingRateDto[]`
```typescript
interface FundingRateDto {
  exchange: string;
  symbol: string;
  rate: number;
  annualizedRate: number;
  fundingTime: string;
  nextFundingTime: string;
  recordedAt: string;
}
```

**Example**:
```typescript
connection.on('ReceiveFundingRates', (data: FundingRateDto[]) => {
  console.log('Funding rates updated:', data);
});
```

**Frequency**: Every 5 seconds (configurable)

---

#### ReceivePositions

Updates on all open positions.

**Event Name**: `ReceivePositions`

**Payload**: `PositionDto[]`
```typescript
interface PositionDto {
  id: number;
  exchange: string;
  symbol: string;
  side: PositionSide;
  status: PositionStatus;
  entryPrice: number;
  exitPrice?: number;
  quantity: number;
  leverage: number;
  initialMargin: number;
  realizedPnL: number;
  unrealizedPnL: number;
  totalFundingFeePaid: number;
  totalFundingFeeReceived: number;
  netFundingFee: number;
  openedAt: string;
  closedAt?: string;
}
```

**Example**:
```typescript
connection.on('ReceivePositions', (data: PositionDto[]) => {
  console.log('Positions updated:', data);
});
```

**Frequency**: Every 5 seconds

---

#### ReceiveOpportunities

New arbitrage opportunities detected.

**Event Name**: `ReceiveOpportunities`

**Payload**: `ArbitrageOpportunityDto[]`
```typescript
interface ArbitrageOpportunityDto {
  id: number;
  symbol: string;
  longExchange: string;
  shortExchange: string;
  longFundingRate: number;
  shortFundingRate: number;
  spreadRate: number;
  annualizedSpread: number;
  estimatedProfitPercentage: number;
  status: OpportunityStatus;
  detectedAt: string;
  executedAt?: string;
}
```

**Example**:
```typescript
connection.on('ReceiveOpportunities', (data: ArbitrageOpportunityDto[]) => {
  console.log('New opportunities:', data);
  // Show notification to user
});
```

**Frequency**: Every 5 seconds (only when opportunities exist)

---

#### ReceiveBalances

Account balance updates from all exchanges.

**Event Name**: `ReceiveBalances`

**Payload**: `AccountBalanceDto[]`
```typescript
interface AccountBalanceDto {
  exchange: string;
  totalBalance: number;
  availableBalance: number;
  marginUsed: number;
  unrealizedPnL: number;
  updatedAt: string;
}
```

**Example**:
```typescript
connection.on('ReceiveBalances', (data: AccountBalanceDto[]) => {
  console.log('Balances updated:', data);
});
```

**Frequency**: Every 5 seconds

---

#### ReceivePnLUpdate

Portfolio-wide P&L summary.

**Event Name**: `ReceivePnLUpdate`

**Payload**:
```typescript
{
  totalPnL: number;
  todayPnL: number;
}
```

**Example**:
```typescript
connection.on('ReceivePnLUpdate', (data) => {
  console.log(`Total P&L: $${data.totalPnL}`);
  console.log(`Today P&L: $${data.todayPnL}`);
});
```

**Frequency**: Every 5 seconds

---

#### ReceiveAlert

System alerts and notifications.

**Event Name**: `ReceiveAlert`

**Payload**:
```typescript
{
  message: string;
  severity: string;
  timestamp: string;
}
```

**Severity Levels**:
- `info`: Informational message
- `warning`: Warning condition
- `error`: Error occurred
- `critical`: Critical issue requiring immediate attention

**Example**:
```typescript
connection.on('ReceiveAlert', (data) => {
  console.log(`[${data.severity}] ${data.message}`);
  // Show toast notification
});
```

**Frequency**: Ad-hoc (as events occur)

---

### Connection Management

#### Reconnection

SignalR automatically reconnects with exponential backoff:
- Initial retry: 0s
- Retry 1: 2s
- Retry 2: 10s
- Retry 3: 30s
- Max: 30s between retries

**Handling Reconnection**:
```typescript
connection.onreconnecting((error) => {
  console.log('Connection lost. Reconnecting...', error);
  // Show UI indicator
});

connection.onreconnected((connectionId) => {
  console.log('Connection restored:', connectionId);
  // Hide UI indicator
});

connection.onclose((error) => {
  console.log('Connection closed:', error);
  // Attempt manual reconnection
  setTimeout(() => connection.start(), 5000);
});
```

---

## Error Responses

### Standard Error Format

```json
{
  "error": "Error message",
  "statusCode": 400,
  "details": {
    "field": "validation error details"
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 204 | No Content (success, no body) |
| 400 | Bad Request (validation error) |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict (duplicate entry) |
| 422 | Unprocessable Entity |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Common Error Scenarios

#### Validation Error (400)
```json
{
  "error": "Validation failed",
  "statusCode": 400,
  "details": {
    "apiKey": "API key is required",
    "leverage": "Leverage must be between 1 and 20"
  }
}
```

#### Not Found (404)
```json
{
  "error": "Exchange with ID 999 not found",
  "statusCode": 404
}
```

#### Exchange API Error (503)
```json
{
  "error": "Unable to connect to Binance API",
  "statusCode": 503,
  "details": {
    "exchange": "Binance",
    "originalError": "Connection timeout"
  }
}
```

---

## Rate Limiting

Currently, rate limiting is not enforced. For production, implement rate limiting:

**Recommended Limits**:
- Read endpoints: 100 requests/minute
- Write endpoints: 20 requests/minute
- SignalR connections: 10 concurrent per IP

**Future Header Format**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642262400
```

---

## API Versioning

Current version: **v1** (implicit)

Future versions will use URL versioning:
- v1: `/api/v1/position`
- v2: `/api/v2/position`

---

## OpenAPI / Swagger

**Swagger UI**: `http://localhost:5000/swagger`

Interactive API documentation available in development mode.

**OpenAPI JSON**: `http://localhost:5000/swagger/v1/swagger.json`

---

## Code Examples

### C# (.NET Client)

```csharp
using System.Net.Http.Json;

var client = new HttpClient
{
    BaseAddress = new Uri("http://localhost:5000")
};

// Get all exchanges
var exchanges = await client.GetFromJsonAsync<List<Exchange>>("/api/exchange");

// Get active opportunities
var opportunities = await client.GetFromJsonAsync<List<ArbitrageOpportunity>>(
    "/api/opportunity/active");

// Update exchange
var response = await client.PutAsJsonAsync("/api/exchange/1", updatedExchange);
```

---

### Python

```python
import requests

BASE_URL = "http://localhost:5000"

# Get all positions
response = requests.get(f"{BASE_URL}/api/position")
positions = response.json()

# Get active opportunities
response = requests.get(f"{BASE_URL}/api/opportunity/active")
opportunities = response.json()

# Toggle exchange
response = requests.post(f"{BASE_URL}/api/exchange/1/toggle")
result = response.json()
print(f"Exchange enabled: {result['enabled']}")
```

---

### JavaScript/TypeScript (with Fetch)

```typescript
const BASE_URL = 'http://localhost:5000';

// Get all exchanges
const response = await fetch(`${BASE_URL}/api/exchange`);
const exchanges = await response.json();

// Update exchange
const updateResponse = await fetch(`${BASE_URL}/api/exchange/1`, {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(updatedExchange),
});

// Get positions filtered by status
const positionsResponse = await fetch(
  `${BASE_URL}/api/position?status=Open`
);
const positions = await positionsResponse.json();
```

---

### curl Examples

```bash
# Get all opportunities with limit
curl -X GET "http://localhost:5000/api/opportunity?limit=10"

# Update exchange configuration
curl -X PUT "http://localhost:5000/api/exchange/1" \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "name": "Binance",
    "apiKey": "test-key",
    "apiSecret": "test-secret",
    "isEnabled": true
  }'

# Toggle exchange
curl -X POST "http://localhost:5000/api/exchange/2/toggle"

# Get open positions only
curl -X GET "http://localhost:5000/api/position?status=Open"
```

---

## Webhooks (Future Feature)

Subscribe to events via webhooks:

```json
POST /api/webhook/subscribe
{
  "url": "https://your-server.com/webhook",
  "events": ["opportunity.detected", "position.closed"],
  "secret": "your-webhook-secret"
}
```

Payload will be signed with HMAC-SHA256.

---

## GraphQL API (Future Feature)

GraphQL endpoint for flexible queries:

```graphql
query {
  opportunities(status: DETECTED, limit: 10) {
    symbol
    spreadRate
    annualizedSpread
    longExchange
    shortExchange
  }

  positions(status: OPEN) {
    symbol
    side
    unrealizedPnL
    netFundingFee
  }
}
```

---

## API Best Practices

### Pagination

For large result sets, use pagination:

```
GET /api/position?page=1&pageSize=20
```

Response includes metadata:
```json
{
  "data": [...],
  "pagination": {
    "currentPage": 1,
    "pageSize": 20,
    "totalPages": 5,
    "totalCount": 100
  }
}
```

### Filtering

Use query parameters for filtering:

```
GET /api/opportunity?symbol=BTCUSDT&minSpread=0.5
```

### Sorting

Use `orderBy` and `direction`:

```
GET /api/position?orderBy=unrealizedPnL&direction=desc
```

### Field Selection

Request specific fields only:

```
GET /api/position?fields=id,symbol,unrealizedPnL
```

---

## Support & Feedback

For API issues or feature requests, please open an issue on GitHub or contact the development team.

**GitHub Repository**: [Link to repository]
