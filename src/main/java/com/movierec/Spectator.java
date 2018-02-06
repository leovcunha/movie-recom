package com.movierec;

import java.util.ArrayList;
import java.util.HashMap;

import javax.persistence.ElementCollection;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;

import lombok.Data;

@Data
@Entity
public class Spectator {

	private @Id @GeneratedValue Long id;
	private String userName;
	@ElementCollection	private HashMap<String, Integer> movieRatings; // movie id-> rating 
    @ElementCollection private ArrayList<String> recommendations; //movie ids
    
	private Spectator() {}

	public Spectator(String userName) {
		this.userName = userName;
		this.movieRatings=  new HashMap<String, Integer>();
		this.recommendations = new ArrayList<String>()
	}
}
