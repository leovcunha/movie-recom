package com.movierec;

import org.springframework.data.jpa.repository.JpaRepository;

public interface SpectatorRepository extends JpaRepository<Spectator, Long> {
	
	Spectator findByUsername(String username);

}