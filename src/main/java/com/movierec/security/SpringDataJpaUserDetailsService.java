/**
 * 
 */
package com.movierec.security;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Component;

import com.movierec.Spectator;
import com.movierec.SpectatorRepository;

/**
 * @author leovcunha
 *
 */
@Component
public class SpringDataJpaUserDetailsService implements UserDetailsService {

	/* (non-Javadoc)
	 * @see org.springframework.security.core.userdetails.UserDetailsService#loadUserByUsername(java.lang.String)
	 */
	
	private final SpectatorRepository repository;
	
	@Autowired
	public SpringDataJpaUserDetailsService(SpectatorRepository repository) {
		this.repository = repository;
	}
	
	@Override
	public UserDetails loadUserByUsername(String name) throws UsernameNotFoundException {
		
		Spectator spectator = this.repository.findByUserName(name);
		return new User(spectator.getUserName(), spectator.getPassword(), null);
	}

}
