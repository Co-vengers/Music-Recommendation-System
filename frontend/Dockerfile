# ---------- Build Stage ----------
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files first (better caching)
COPY package.json package-lock.json ./

RUN npm install

# Copy rest of the source
COPY . .

# Build React app
RUN npm run build


# ---------- Production Stage ----------
FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy build output from previous stage
COPY --from=build /app/build /usr/share/nginx/html

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
