# Базовий образ для node
FROM node:18-alpine

WORKDIR /app

# Копіюємо package.json та package-lock.json (якщо є)
COPY package*.json ./

# Встановлюємо залежності
RUN npm install

# Копіюємо увесь код фронтенду
COPY . .

# Будуємо продакшен-бандл
RUN npm run build

# Використовуємо nginx для роздачі готового фронтенду
FROM nginx:stable-alpine

# Копіюємо збірку з першого етапу в папку nginx
COPY --from=0 /app/build /usr/share/nginx/html

# Відкриваємо 80 порт
EXPOSE 80

# Запускаємо nginx
CMD ["nginx", "-g", "daemon off;"]
