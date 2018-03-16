package com.movierec;

import org.springframework.data.repository.CrudRepository;

public interface SpectatorRepository extends CrudRepository<Spectator, Long> {
	
	Spectator findByUserName(String userName);

}