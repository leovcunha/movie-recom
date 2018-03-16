package com.movierec;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

import javax.persistence.ElementCollection;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;


import javax.validation.constraints.NotNull;

import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

import com.fasterxml.jackson.annotation.JsonIgnore;

import lombok.Data;
import lombok.ToString;

@Data
@Entity
@ToString(exclude = "password")
public class Spectator {
	
	public static final PasswordEncoder PASSWORD_ENCODER = new BCryptPasswordEncoder();

	private @Id @GeneratedValue Long id;
	private @NotNull String userName;
	@ElementCollection	private Map<String, Integer> movieRatings; // movie id-> rating 
    @ElementCollection private List<String> recommendations; //movie ids * Must be declared as interface
    private @JsonIgnore String password;
    
    public void setPassword(String password) {
    		this.password =PASSWORD_ENCODER.encode(password);
    }
    
	private Spectator() {}

	public Spectator(String userName, String password) {
		this.userName = userName;
		this.setPassword(password);
		this.movieRatings=  new HashMap<String, Integer>();
		this.recommendations = new ArrayList<String>();
	}
}
