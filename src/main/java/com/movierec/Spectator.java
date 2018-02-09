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

import lombok.Data;

@Data
@Entity
public class Spectator {

	private @Id @GeneratedValue Long id;
	private @NotNull String userName;
	@ElementCollection	private Map<String, Integer> movieRatings; // movie id-> rating 
    @ElementCollection private List<String> recommendations; //movie ids * Must be declared as interface
    
	private Spectator() {}

	public Spectator(String userName) {
		this.userName = userName;
		this.movieRatings=  new HashMap<String, Integer>();
		this.recommendations = new ArrayList<String>();
	}
}
