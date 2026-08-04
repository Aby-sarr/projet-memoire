package com.wolof.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Configuration CORS (Cross-Origin Resource Sharing).
 *
 * Sans cette config, le navigateur bloque les requêtes Angular (port 4200)
 * vers Spring Boot (port 8080) car ils ont des origines différentes.
 *
 * Cette classe autorise Angular à appeler l'API Spring Boot.
 */
@Configuration
public class CorsConfig {

    // Valeur lue depuis application.yml
    @Value("${app.cors.allowed-origins}")
    private String allowedOrigins;

    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry
                    .addMapping("/api/**")           // Toutes les routes /api/*
                    .allowedOrigins(allowedOrigins)  // http://localhost:4200
                    .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                    .allowedHeaders("*")
                    .allowCredentials(true);
            }
        };
    }
}
