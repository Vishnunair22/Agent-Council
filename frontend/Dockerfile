FROM node:20-alpine

WORKDIR /app

# Copy package locks first
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Define public env vars
ENV NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
EXPOSE 3000

# Start in development mode so user can see live changes
CMD ["npm", "run", "dev"]
