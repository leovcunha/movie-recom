/**
 * 
 */
package com.movierec.security;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.authority.AuthorityUtils;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import static java.util.Collections.emptyList;

import com.movierec.Spectator;
import com.movierec.SpectatorRepository;

/**
 * @author leovcunha
 *
 */
@Service
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
		
		Spectator spectator = this.repository.findByUsername(name);

        if (spectator == null)
           throw new UsernameNotFoundException("User not found");
    		
    	return new User(spectator.getUsername(), spectator.getPassword(), emptyList());
	}

}
