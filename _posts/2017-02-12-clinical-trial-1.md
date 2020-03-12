---                                                                                       
layout: post
title: "Building a Clinical Trial Simulator (Part I)"
description: "We use C++ to create a clinical trial simulator."
comments: true
keywords: "c++, cpp, clinical trial, randomized, RCT, pre-exposure prophylaxis, PrEP, 
HIV, AIDS, HIV/AIDS, human immunodeficiency virus, acquired immune deficiency syndrome,
simulation"
---  

Let's build a clinical trial simulator! Why might we want to do this? Well, a few possible reasons:

1. clinical trials are expensive, simulators are not
2. you can test new strategies and drugs before conducting an actual clinical trial
3. it might be hard to sample from your test population
4. maybe you are bored

Okay, so maybe a general trial simulator is a bit too generic, since there are many different kinds of 
clinical trials. Maybe I should be more specific. We will create a simple [randomized controlled trial](https://en.wikipedia.org/wiki/Randomized_controlled_trial)
to test efficacy of an HIV prevention intervention, specifically a pre-exposure prophylaxis (PrEP) drug such as
[emtricitabine/tenofovir disoproxil fumarate](https://en.wikipedia.org/wiki/Emtricitabine/tenofovir) 
which is sold under the brand name [TRUVADA &reg;](http://www.truvada.com/). We will have only two arms: 
a control arm and an active arm. Since we don't have any ethics concerns (this is a computer 
simulation after all), we will give the control arm a placebo and the active arm PrEP. Finally, we need
to decide on a test population. Let's consider the sexually active, uninfected, female population
in South Africa, aged 15-49.

We will be using C++ to create the simulator. You can find the code that goes along with this
post [here](https://github.com/danielwo/clinical-trial). For this project I am using Microsoft's compiler:

~~~
Microsoft (R) C/C++ Optimizing Compiler Version 19.00.24215.1 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.
~~~

I will also be using the Gregorian datetime library from boost version 1.61.0 in order to iterate daily and monthly
activity throughout the year. However, you can just assume 30 days in a month and 360 days in a year, and just iterate
with a simple for loop. If you are using a different compiler, I am only going to be using the standard library and 
the above boost library, so you shouldn't have many problems.

First, we are going to spend some time actually designing a model of a clinical trial so that we will know what
we need to implement codewise. Our design choices will affect the complexity and difficulty of implementation. We will
try to simplify when we can. A differential equation model probably won't be a good idea, because we will be dealing with a small population,
say 2000 participants, so the continuity assumptions don't make sense. We could also try a network model, and this is probably a good
idea. However, we would need to find ways to inform the network structure and parameterization can be difficult.
Let's make a compromise. We will consider participants as individual nodes with possible edge attachments to 
partners in the general population, but we will model the general population as a homogenous constant mixture.
Imagine an atom floating around randomly in a well mixed solution looking to pick up some extra electrons, where the probability
of encountering other atoms to form bonds with is relatively low. We will run the trial as a Monte Carlo simulation.

So before we start getting too crazy, let's create a class to store our random number generator (RNG) and sample some basic probability
distributions we will need. We will need a uniform distribution as the workhorse for our general decision making which we will
do via a Bernoulli distribution. We will want to be able to sample from a general discrete distribution for decisions with more than
two outcomes, and finally we will want to be able to sample integers in a range for dealing with properties involving a random
number of days.

~~~cpp
/* RNG.h */
#pragma once

#include <random> // used for probability distributions and RNG

#include <memory> // used for shared pointer


class RNG {
public:
	// constructors
	RNG(std::mt19937& generator);

	//
	double SampleUniformDistribution();
	bool SampleBernoulliDistribution(const double& probability);
	int SampleDiscreteDistribution(std::discrete_distribution<int>& discreteDistribution);
	int SampleIntDistribution(std::uniform_int_distribution<int>& intDistribution);
	double DetermineProbability(const double& baseProbability);
private:
	std::shared_ptr<std::mt19937> m_generator;
	std::uniform_real_distribution<double> m_uniform; // this is the only distribution which is fixed at the start
	
};
~~~

If we find we need access to more distributions, it should be easy to simply add them here.

~~~cpp
/* RNG.cpp */
#include "RNG.h"


// constructors
RNG::RNG(std::mt19937& generator)
	: m_generator(std::make_shared<std::mt19937>(generator))
	, m_uniform(std::uniform_real_distribution<double>(0.0, 1.0)) {}

//
double RNG::SampleUniformDistribution() { return m_uniform(*m_generator); }

bool RNG::SampleBernoulliDistribution(const double& probability) {
	if (SampleUniformDistribution() < probability)
		return true;
	else
		return false;
}

int RNG::SampleDiscreteDistribution(std::discrete_distribution<int>& discreteDistribution) {
	return discreteDistribution(*m_generator);
}

int RNG::SampleIntDistribution(std::uniform_int_distribution<int>& intDistribution) {
	return intDistribution(*m_generator);
}

double RNG::DetermineProbability(const double& baseProbability) {
	return baseProbability * (0.5 + SampleUniformDistribution());
}
~~~

You can see I randomly threw in the `DetermineProbability` method. That is just to provide
some randomization for some of our to be parameters. You can change this method or even delete
it if you so choose.

Now let's try to figure out what kinds of things our participants can do which will influence our model. There are many things people do, 
such as going out to eat, watching movies, etc. These might be relevant if we wanted to model every detail of their lives, but let's
go with just the basics. It is often the case that there is a subset of the population which is high-risk in that they engage more often 
in risky situations and have more sexual partnerships. So first, we will split our population into _high-risk_ and _low-risk_ individuals. 
A person in the population can have HIV, and if they have HIV, they can be on antiretroviral treatment. We know that their infectivity is 
also depedent on their stage of infection. We will simplistically divide an HIV infection into three stages based on duration: acute (recently infected), 
asymptomatic (the period after infection when viral load is low), and late (the period when the viral load increases again and the 
person progresses to AIDS). 

Individuals can form _partnerships_ with other humans. We will consider two partnership types: short-term and long-term partnerships.
To characterize high-risk individuals as engaging in more risky behavior, we will assert that high-risk individuals can engage in
up to two partnerships at once as one of the following five possibilities:

1. no partners
2. one short-term partner
3. two short-term partners
4. one long-term partner
5. one short-term and one long-term partner.

On the other hand, we will assume low-risk individuals can have at most one partner at a time:

1. no partners
2. one short-term partner
3. one long-term partner.

Each partnership can have several characteristics. A partnership can be a short- or long-term partnership,
there will be some frequency of sex acts, there will be some frequency of condom use during sex, and there will be some frequency of anal
sex given that a sexual act takes place. This last one is important because there is a greater risk of HIV transmission with anal sex as compared to vaginal sex. 

At this point, it might be nice to separate people into some distinct classes. The individuals we really care about are _participants_ in our trial.
These participants can form _partnerships_ with individuals we will call _partners_. Because of this distinction, I am going to go with
a partially object oriented approach. We will have our female participants interact with a well mixed population of men, and because of this,
we will only model partnerships as forming from the point of view of the participants. We will not keep track of partners aside from the fact that
they may have HIV, they may get infected outside of the relationship, and they may be on ART. That is, we will not keep track of the partners of the partners.

The only things participants and partners will have in common is that they will have a risk group (high or low), they might have HIV, and if they have HIV, they have
a date of infection. The date of infection is important because it will let us calculate the stage of HIV infection. So let's start with the standard _person_ class:

~~~cpp
/* Person.h */
#pragma once

#include "boost/date_time/gregorian/gregorian.hpp" // used for dates


enum RiskGroup { low_risk, high_risk };

class Person {
public:
	// constructors
	Person(RiskGroup riskGroup); // uninfected person
	Person(RiskGroup riskGroup, boost::gregorian::date m_dateOfInfection); // infected person


	// getters
	RiskGroup GetRiskGroup() const;
	bool Person::HasHiv() const;
	boost::gregorian::date GetDateOfInfection() const;

	// setters
	void Person::SetHivStatus(bool hivStatus);
	void SetDateOfInfection(boost::gregorian::date currentDate);
private:
	RiskGroup m_riskGroup;
	bool m_hasHiv;
	boost::gregorian::date m_dateOfInfection;
};
~~~

~~~cpp
/* Person.cpp */
#include "Person.h"


// constructors
Person::Person(RiskGroup riskGroup)
	: m_riskGroup(riskGroup)
	, m_hasHiv(false) {}

Person::Person(RiskGroup riskGroup, boost::gregorian::date dateOfInfection)
	: m_riskGroup(riskGroup)
	, m_hasHiv(true)
	, m_dateOfInfection(dateOfInfection) {}

// getters
RiskGroup Person::GetRiskGroup() const { return m_riskGroup; }
bool Person::HasHiv() const { return m_hasHiv; }
boost::gregorian::date Person::GetDateOfInfection() const { return m_dateOfInfection; }

// setters
void Person::SetDateOfInfection(boost::gregorian::date currentDate) { m_dateOfInfection = currentDate; }
void Person::SetHivStatus(bool hivStatus) { m_hasHiv = hivStatus; }
~~~

Not much to say here. We have separate constructors for people with and without HIV, denoted by whether or not we pass in a date of infection. 
We have getter methods for all the variables, but don't give a setter method for the risk group. This is because we will assume that people don't
change their risk status. Let's model the partner and partnership next. We will save the participant for last. 

For the partner, the only thing we add is that we allow them to be on ART. You could throw this in the person class if you wanted to, but I put it
here because we will remove participants from the trial whenever they become infected, so it won't really matter for them.

~~~cpp
/* Partner.h */
#pragma once

#include "Person.h"


class Partner :
	public Person {
public:
	//constructors
	Partner(RiskGroup riskGroup); // create a partner with no HIV
	Partner(RiskGroup riskGroup, boost::gregorian::date dateOfInfection, bool onArt); // create a partner with HIV

	// getters
	bool OnArt() const;

	// setters
	void SetArtStatus(bool artStatus);

private:
	bool m_onArt;
};
~~~

~~~cpp
/* Partner.cpp */
#include "Partner.h"


// constructors
Partner::Partner(RiskGroup riskGroup)
	: Person(riskGroup) {}

Partner::Partner(RiskGroup riskGroup, boost::gregorian::date dateOfInfection, bool onArt)
	: Person(riskGroup, dateOfInfection)
	, m_onArt(onArt) {}

// getters
bool Partner::OnArt() const { return m_onArt; }

// setters
void Partner::SetArtStatus(bool artStatus) { m_onArt = artStatus; }
~~~

Note here, for the case when the partner has HIV, we also assign their ART status. Okay, so far so good. We haven't been concerned with any
detailed constructors thus far, and that is because the parameters governing their creation depend on the specific trial being implemented. Therefore, we
will throw their construction in whenever we design the trial. This will be a running theme for all of our "people" classes. This will make our trial class
rather involved, when it doesn't have to be. Feel free to choose a different division of labor for your classes!

As noted before we will create partnerships with several characteristics:

1. Partnership Type (short-term or long-term)
2. Date of partnership formation (this will be used to know when to change short-term partnerships into long-term partnerships)
3. Sexual frequency
4. Condom use frequency
5. Anal sex frequency given a sex act takes place

Here, I opt to have the partnership contain a reference to the participant, and act as the container for the partner. This will allow us to create
functions which act on partnerships instead of having to go through either the participant or partner separately. The partner is contained in
the partnership only because we do not consider their existence outside of partnerships with participants. That is, when a partnership ends,
we stop following the partner. Note, we could have condom use depend on type of sexual act, but we will consider it as independent for simplicity.
We will also want to have some control over whether or not the partnership breaks up. You don't necessarily need this variable, but
I added a boolean `breakUp` which will tell the simulation to break the partnership. We'll see this in action when we start discussing
functions which act on the objects we're creating.

~~~cpp
/* Partnership.h */
#pragma once

#include "RNG.h"

#include "Participant.h"

#include "Partner.h"

#include "boost/date_time/gregorian/gregorian.hpp" // used for dates


enum PartnershipType { long_term, short_term };

class Partnership {
public:
	// constructors
	Partnership(Participant& participant, Partner partner, PartnershipType partnershipType, 
                boost::gregorian::date startDate, double sexFrequency, double condomFrequency, 
                double analFrequency);
	
	// getters
	PartnershipType GetPartnershipType() const;
	boost::gregorian::date GetStartDate() const;
	double GetSexFrequency() const;
	double GetCondomFrequency() const;
	double GetAnalFrequency() const;
	Participant& GetParticipantRef() const;
	Partner& GetPartnerRef();
	bool breakUp() const;

	// setters
	void SetPartnershipType(PartnershipType partnershipType);
	void SetSexFrequency(double sexFrequency);
	void SetCondomFrequency(double condomFrequency);
	void SetAnalFrequency(double analFrequency);
	void SetBreakUp(bool breakUp);

private:
	Participant& m_participant;
	Partner m_partner;
	PartnershipType m_partnershipType; // type of partnership (short_term, long_term)
	
	double m_sexFrequency;
	double m_condomFrequency;
	double m_analFrequency; // conditional probability to have anal sex, given sex
	
	boost::gregorian::date m_startDate;
	bool m_breakUp; // true if partnership breaks up
	
};
~~~

~~~cpp
/* Partnership.cpp */
#include "Partnership.h"


// constructors
Partnership::Partnership(Participant& participant, Partner partner, PartnershipType partnershipType, 
        boost::gregorian::date startDate, double sexFrequency, double condomFrequency, 
        double analFrequency)
	: m_participant(participant)
	, m_partner(partner)
	, m_startDate(startDate)
	, m_partnershipType(partnershipType)
	, m_sexFrequency(sexFrequency)
	, m_condomFrequency(condomFrequency)
	, m_analFrequency(analFrequency)
	, m_breakUp(false) {}

// getters
PartnershipType Partnership::GetPartnershipType() const { return m_partnershipType; }
boost::gregorian::date Partnership::GetStartDate() const { return m_startDate; }
double Partnership::GetSexFrequency() const { return m_sexFrequency; }
double Partnership::GetCondomFrequency() const { return m_condomFrequency; }
double Partnership::GetAnalFrequency() const { return m_analFrequency; }
Participant& Partnership::GetParticipantRef() const { return m_participant; }
Partner& Partnership::GetPartnerRef() { return m_partner; }
bool Partnership::breakUp() const { return m_breakUp; }

// setters
void Partnership::SetPartnershipType(PartnershipType partnershipType) { m_partnershipType = partnershipType; }
void Partnership::SetSexFrequency(double sexFrequency) { m_sexFrequency = sexFrequency; }
void Partnership::SetCondomFrequency(double condomFrequency) { m_condomFrequency = condomFrequency; }
void Partnership::SetAnalFrequency(double analFrequency) { m_analFrequency = analFrequency; }
void Partnership::SetBreakUp(bool breakUp) { m_breakUp = breakUp; }
~~~

You may have noticed that many of these classes so far have only been comprised of getter and setter methods. Really, you may want to
consider just creating vectors or arrays which store these attributes and then act on them. This will result in faster
simulations, but I figured these classes made sense as actual objects.

Okay, now finally we'll create a participant class! For this, we will have to think a tiny bit about how the actual trial works, 
which we haven't thought about very much yet. We will also have to explore how we want to store the partnership concurrency status of the
participant.

Another issue is that often participants aren't all enrolled at the same time, but staggered monthly as the trial goes on, 
and the participants are then followed. We could consider a fixed length trial design, where all participants will be followed for a 
fixed amount of time or until infection. In this case, from the standpoint of the simulation, it doesn't matter if they are enrolled 
all at once or in a stagged manner. However, for this example I decided on an event-driven trial structure. That is, we will follow 
the participants until a threshhold number of infections have been observed, at which point the trial ends. In this case, 
as you might imagine, the staggered enrollment matters. So we will give each participant an enrollment status 
(initially false) and an enrollment date at which point they become enrolled. Then we will give the participants 
a maximum follow up date, at which point if they haven't been infected or if the trial hasn't ended, we stop 
following them. 

Also we can enroll participants in either the active arm (where they take PrEP) or in 
the control arm (where they take a placebo). For this we will create an Arm enum:

~~~cpp
enum Arm { control, active }; // arm of the trial the participant is enrolled in
~~~

If you remember, we had decided that participants have one out of five states of partnership concurrency:

1. no partners (_0L0S)
2. one short-term partner (_0L1S)
3. two short-term partners (_0L2S)
4. one long-term partner (_1L0S)
5. one short-term and one long-term partner (_1L1S).

For this we will create a Concurrency enum:

~~~cpp
enum Concurrency { _0L0S, _0L1S, _0L2S, _1L0S, _1L1S }; // number of partners 
				// _xLyS  means x long_term partners and y short_term partners
~~~

Whenever the participant gains or loses a partner (through some function which creates or destroys partnerships), or when a partner
turns from a short-term to a long-term partner, we want to be able to change the participant's concurrency status. To do this we can actually be a little clever with some arithmetic:

~~~cpp
enum class ConcurrencyChange  { // this will allow switching between different concurrency states easily
	none = 0,
	new_short_term = 1,
	new_long_term = 3,
	break_short_term = -1,
	break_long_term = -3,
	update_short_term_to_long_term = 2
};
~~~

Defining the enum class ConcurrencyChange, we can then just add the proposed "change in concurrency" to the current 
concurrency, e.g. _0L1S + new_long_term = 1 + 3 = 4 = _1L1S. We just have to make sure we don't give any invalid
changes.

~~~cpp
/* Participant.h */
#pragma once

#include "Person.h"

#include "boost/date_time/gregorian/gregorian.hpp" // used for dates


enum Arm { control, active }; // arm of the trial the participant is enrolled in
enum Concurrency { _0L0S, _0L1S, _0L2S, _1L0S, _1L1S }; // number of partners 
				// _xLyS  means x long_term partners and y short_term partners
				
enum class ConcurrencyChange  { // this will allow switching between different concurrency states easily
	none = 0,
	new_short_term = 1,
	new_long_term = 3,
	break_short_term = -1,
	break_long_term = -3,
	update_short_term_to_long_term = 2
};

class Participant :
	public Person {
public:
	// constructors
	Participant(RiskGroup riskGroup, Arm trialArm, boost::gregorian::date enrollmentDate, 
        boost::gregorian::date maxFollowUpDate, Concurrency concurrencyStatus);  // enrolled participant


	// getters
	Concurrency GetConcurrencyStatus() const;
	Arm GetTrialArm() const;
	boost::gregorian::date GetMaxFollowUpDate() const;
	boost::gregorian::date GetEnrollmentDate() const;
	bool IsEnrolled() const;

	// setters
	void SetTrialArm(Arm trialArm);
	void SetEnrollmentDate(boost::gregorian::date enrollmentDate);
	void SetEnrollmentStatus(bool enrollmentStatus);

	//
	void UpdateConcurrencyStatus(ConcurrencyChange concurrencyChange);
private:
	boost::gregorian::date m_dateOfInfection;
	Concurrency m_concurrencyStatus;
	Arm m_trialArm;
	bool m_isEnrolled;
	boost::gregorian::date m_enrollmentDate;
	boost::gregorian::date m_maxFollowUpDate;
};
~~~

~~~cpp
/* Participant.cpp */
#include "Participant.h"


// constructors
Participant::Participant(RiskGroup riskGroup, Arm trialArm, boost::gregorian::date enrollmentDate, 
        boost::gregorian::date maxFollowUpDate, Concurrency concurrencyStatus)
	: Person(riskGroup) // we enroll healthy participants only
	, m_trialArm(trialArm)
	, m_isEnrolled(false) // we only enroll participants at their enrollment date
	, m_enrollmentDate(enrollmentDate)
	, m_maxFollowUpDate(maxFollowUpDate)
	, m_concurrencyStatus(concurrencyStatus) {}

// getters
Concurrency Participant::GetConcurrencyStatus() const { return m_concurrencyStatus; }
Arm Participant::GetTrialArm() const { return m_trialArm; }
boost::gregorian::date Participant::GetEnrollmentDate() const { return m_enrollmentDate; }
boost::gregorian::date Participant::GetMaxFollowUpDate() const { return m_maxFollowUpDate; }
bool Participant::IsEnrolled() const { return m_isEnrolled; }

// setters
void Participant::SetEnrollmentStatus(bool enrollmentStatus) { m_isEnrolled = enrollmentStatus; }
void Participant::SetTrialArm(Arm trialArm) { m_trialArm = trialArm; }
void Participant::SetEnrollmentDate(boost::gregorian::date enrollmentDate) { m_enrollmentDate = enrollmentDate; }

//
void Participant::UpdateConcurrencyStatus(ConcurrencyChange concurrencyChange) {
	int newConcurrency = m_concurrencyStatus + static_cast<int>(concurrencyChange);
	if (0 <= newConcurrency && newConcurrency <= 4)
		m_concurrencyStatus = static_cast<Concurrency>(newConcurrency);
	else
	{
		std::cerr << "Not a valid concurrency change!" << std::endl;
	}
}
~~~

Alright, all together this has been a lot to go through. Let's finish off with an example of an rng-participant-partner-partnership interaction.
Here, we first initialize the participant as a low-risk individual in the control arm with no partners.
We then create an HIV infected, high-risk partner. We set some dummy variables for sexual frequency,
condom frequency and anal frequency. We create a new partnership and update the participant's
concurrency status. We set up some more dummy variables for condom efficacy, ART efficacy 
and the effect of anal sex on HIV transmission. We then use the RNG class we created
to simulate a sexual encounter: 

~~~cpp
/* example_1.cpp */

#include <iostream>

#include <memory>

#include "RNG.h"

#include "Partnership.h"


int main() {
	boost::gregorian::date startDate = boost::gregorian::date(2017, boost::gregorian::Feb, 2); // start of trial
	
	boost::gregorian::date maxFollowDate = startDate + boost::gregorian::months(12); 
    
	// enrollment date is start date, better enroll the participant!
	Participant my_participant(low_risk, control, startDate, maxFollowDate, _0L0S); 
	my_participant.SetEnrollmentStatus(true);
    
	// infected 3 months ago!! (we aren't using this yet)
	boost::gregorian::date dateOfInfection = startDate - boost::gregorian::months(3); 
	Partner my_partner(high_risk, dateOfInfection, false); // infected but not on ART
	
	double sexFrequency = 1;
	double condomFrequency = 0.1;
	double analFrequency = 0.3;
	
	std::unique_ptr<Partnership> new_partnership =  std::make_unique<Partnership>(my_participant, 
                my_partner, short_term, startDate, sexFrequency, condomFrequency, analFrequency);

    	// better update the concurrency status of the participant!
    	my_participant.UpdateConcurrencyStatus(ConcurrencyChange::new_short_term);

	// initialize HIV/sex related parameters
	double condomEfficacy = 0.9; // 90% effectiveness in preventing transmission
	double artEffiacy = 0.8; // efficacy of antiretroviral
	double analMultiplier = 5; // 5x infectiousness on anal intercourse


	// set up simple seed with mersenne twister RNG
	// std::mt19937 my_rng(std::random_device{}());
	unsigned seed = 3;
	std::mt19937 my_generator(seed); 
	RNG my_rng(my_generator);

	/*********************************************************************/
	// partnership sex
	boost::gregorian::date today = startDate; // date of sexual interaction
	
	Participant& current_participant = new_partnership->GetParticipantRef();
	Partner& current_partner = new_partnership->GetPartnerRef();

	double hivTransmissionProbability; 
	if (current_partner.HasHiv()) {
		hivTransmissionProbability = 0.15; // 15% base transmission probability (not realistic)

		std::cout << "Partner has HIV!" << std::endl;
		std::cout << "hivTransmissionProbability = " << hivTransmissionProbability << std::endl;
	}

	// if the partner does not have HIV, no reason to have sex. Can save computing time here!
	bool haveSex = my_rng.SampleBernoulliDistribution(new_partnership->GetSexFrequency());
	if (haveSex) {
		std::cout << "Partnership has sex!" << std::endl;

		if (current_partner.OnArt()) {
			hivTransmissionProbability *= (1 - artEffiacy); // reduce infectiousness
			
			std::cout << "Partner is on ART!" << std::endl;
			std::cout << "hivTransmissionProbability = " << hivTransmissionProbability << std::endl;
		}

		bool useCondom = my_rng.SampleBernoulliDistribution(new_partnership->GetCondomFrequency());
		if (useCondom) {
			hivTransmissionProbability *= (1-condomEfficacy); // reduce infectiousness
			
			std::cout << "Condom is used!" << std::endl;
			std::cout << "hivTransmissionProbability = " << hivTransmissionProbability << std::endl;
		}

		bool haveAnalSex = my_rng.SampleBernoulliDistribution(new_partnership->GetAnalFrequency());
		if (haveAnalSex) {
			hivTransmissionProbability *= analMultiplier; // increase infectiousness
			
			std::cout << "Partnership has anal sex!" << std::endl;
			std::cout << "hivTransmissionProbability = " << hivTransmissionProbability << std::endl;
		}

		bool hivTransmitted = my_rng.SampleBernoulliDistribution(hivTransmissionProbability);
		if (hivTransmitted) {
			current_participant.SetHivStatus(true);
			current_participant.SetDateOfInfection(today);
			std::cout << "HIV has been transmitted!" << std::endl;
		}
	}
}
~~~

~~~
Output
------
Partner has HIV!
hivTransmissionProbability = 0.15
Partnership has sex!
Partnership has anal sex!
hivTransmissionProbability = 0.75
HIV has been transmitted!
~~~

Okay, now we have some basic classes set up which we can use in a simulation. We will put all
the pieces together in a full trial simulation in part II!
