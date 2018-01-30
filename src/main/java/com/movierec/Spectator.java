package com.movierec;

import java.util.HashMap;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;

import lombok.Data;

@Data
@Entity
public class Spectator {

	private @Id @GeneratedValue Long id;
	private String userName;
	private HashMap<String, Integer> movieRatings;

	private Spectator() {}

	public Spectator(String userName,  HashMap<String, Integer> ratings) {
		this.userName = userName;
		this.movieRatings= movieRatings;
	}
}
