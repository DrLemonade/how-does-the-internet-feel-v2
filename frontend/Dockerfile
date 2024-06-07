FROM node:16.14.0 as builder
WORKDIR /usr/src/app
COPY package.json yarn.lock ./

RUN yarn

COPY . ./
RUN yarn build

FROM nginx:1.23.4-alpine

COPY ./.nginx/nginx.conf /etc/nginx/nginx.conf

RUN rm -rf /usr/share/nginx/html/*

COPY --from=builder /usr/src/app/dist /usr/share/nginx/html
