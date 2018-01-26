package com.movierec;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;

import lombok.Data;

@Data
@Entity
public class Spectator {

	private @Id @GeneratedValue Long id;
	private String username;
	private String description;

	private Spectator() {}

	public Spectator(String username, String description) {
		this.username = username;
		this.description = description;
	}
}
