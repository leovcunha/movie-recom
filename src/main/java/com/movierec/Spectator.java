package com.movierec;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

import javax.persistence.Column;
import javax.persistence.ElementCollection;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.validation.constraints.NotNull;

import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

import com.fasterxml.jackson.annotation.JsonIgnore;

import com.movierec.recommender.MovieRecommender;

import lombok.Data;
import lombok.ToString;

@Data
@Entity
@ToString(exclude = "password")
public class Spectator implements Serializable {
	
	public static final PasswordEncoder PASSWORD_ENCODER = new BCryptPasswordEncoder();
    private static final long serialVersionUID = 1L;
	private @Id @GeneratedValue Long id;
	private @NotNull String username;
	@ElementCollection	private Map<Long, Float> movieratings; // movie id-> rating 
    @ElementCollection private Map<Long, Float> recommendations; //movie ids * Must be declared as interface
    private @Column(length = 60) @NotNull String password;

	private Spectator() {}

    public void setPassword(String password) {
        
        this.password = PASSWORD_ENCODER.encode(password);
        
    }

	public Spectator(String uName, String pword) {
			this.username = uName;
			this.setPassword(pword);

		this.movieratings=  new HashMap<Long, Float>();
		this.recommendations = new HashMap<Long, Float>();
	}
}
