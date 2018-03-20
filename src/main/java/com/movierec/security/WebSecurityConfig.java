package com.movierec.security;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;

import com.movierec.Spectator;

@Configuration
@EnableWebSecurity
public class WebSecurityConfig extends WebSecurityConfigurerAdapter {
    
	@Autowired
	private SpringDataJpaUserDetailsService userDetailsService;
	
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
            .authorizeRequests()
                .antMatchers("/", "/built/**", "/css/**").permitAll()
               // .anyRequest().fullyAuthenticated()
                .and()
            .formLogin()
     
                .defaultSuccessUrl("/", true)
                .permitAll()
                .and()
            .logout()
                .permitAll()
        			.and()
        		.csrf().disable();       
    }
    
    @Autowired
    public void configure(AuthenticationManagerBuilder auth) throws Exception {
        auth.userDetailsService(this.userDetailsService)
        			.passwordEncoder(Spectator.PASSWORD_ENCODER);        
    }

}
