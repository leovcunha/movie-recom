package com.movierec.security;

import com.movierec.Spectator;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;


import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;

import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Date;

import static com.movierec.security.SecurityConstants.EXPIRATION_TIME;
import static com.movierec.security.SecurityConstants.HEADER_STRING;
import static com.movierec.security.SecurityConstants.SECRET;
import static com.movierec.security.SecurityConstants.TOKEN_PREFIX;

/**
 * Adds an authentication filter that checks username and password and if correct sends back a JsonwebToken
 * to be used with Authorization Filter
 */
public class JWTAuthenticationFilter extends UsernamePasswordAuthenticationFilter {
    private AuthenticationManager authenticationManager;

    public JWTAuthenticationFilter(AuthenticationManager authenticationManager) {
        this.authenticationManager = authenticationManager;
    }

/**
 * Checks username & password with authentication manager injected by WebSecurity 
 */
    @Override
    public Authentication attemptAuthentication(HttpServletRequest req,
                                                HttpServletResponse res) throws AuthenticationException {
        try {
            Map<String, String> creds = new ObjectMapper()
                    .readValue(req.getInputStream(), new TypeReference<Map<String,String>>(){});


            return authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(
                            creds.get("username"), //if create spectator instance auth will fail because
                            creds.get("password"), //of encoder is inside model. Could have changed if wanted this design.
                            new ArrayList<>())
                            
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
/**
 * Create token and send to user in header with prefix
 */
    @Override
    protected void successfulAuthentication(HttpServletRequest req,
                                            HttpServletResponse res,
                                            FilterChain chain,
                                            Authentication auth) throws IOException, ServletException {

        String token = Jwts.builder()
                .setSubject(((User) auth.getPrincipal()).getUsername())
                .setExpiration(new Date(System.currentTimeMillis() + EXPIRATION_TIME))
                .signWith(SignatureAlgorithm.HS512, SECRET.getBytes())
                .compact();
        res.addHeader(HEADER_STRING, TOKEN_PREFIX + token);
    }
}