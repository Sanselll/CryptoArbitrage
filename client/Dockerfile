# Multi-stage Dockerfile for React Frontend

# Stage 1: Build
FROM node:20-alpine AS build
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (including dev for build)
RUN npm ci

# Copy source code
COPY . .

# Accept build arguments
ARG VITE_GOOGLE_CLIENT_ID
ARG VITE_API_BASE_URL_DEMO
ARG VITE_API_BASE_URL_REAL

# Set environment variables for build
ENV VITE_GOOGLE_CLIENT_ID=${VITE_GOOGLE_CLIENT_ID}
ENV VITE_API_BASE_URL_DEMO=${VITE_API_BASE_URL_DEMO}
ENV VITE_API_BASE_URL_REAL=${VITE_API_BASE_URL_REAL}

# Build the application
RUN npm run build

# Stage 2: Production with Nginx
FROM nginx:alpine AS final

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
