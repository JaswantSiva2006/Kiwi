Product Vision: 

What really drew me to this task in the first place was the long-term vision behind Kiwi. With its emphasis on ease of use and seamless integration, I genuinely believe Kiwi has the potential to become something massive.

When I picture where this could go, I think about how effortless human-computer interaction should actually feel. In an ideal setup, you open your laptop and talk to Kiwi the way Tony Stark talks to JARVIS. You say, "Fetch me everything on my desk for today," and instantly, your scatter of emails, task boards, and messages are gathered, clean, and ranked by actual priority.

The real leap happens when we solve deep cross-app connectivity. Right now, most software stays locked inside digital communication, but taking Kiwi to the next level means connecting it to real life. It means Kiwi understanding user context deeply enough (i mean it should be able to understand the evolving world of the user) to handle real world tasks like ordering your lunch during a back-to-back meeting day, or automatically booking a cab the second you wrap up work and close your laptop.

Developing this task properly is really about teaching Kiwi the semantics of work like how different apps, actions, and user habits relate to one another (thats why i wana focus on building the HNSW graph as well apart from just a memory ledger, relationship between memories imo is very important). Building a true JARVIS comes down to bringing four core pieces together:

Semantics: Deeply understanding workflow relationships and user intent.
I thought about this and eventually ended up with the conclusion that the to most important things I would want the semantics to remember ia (A) relationships (every subject object relation is important for us to build the graph which will help us far distant memories which might not be semantically same but still might be needed during retrieval to answer efficiently. (b) Temporalness: We absolutely should make it temporally aware of every event in the user’s life and this is what my architecture primarily focuses on)

App Connectivity: Executing actions across third-party software seamlessly.

Voice: Keeping input natural, fast, and friction-free. (I think Sarvam has kind of aced it here already)

Compute: Running real-time context processing without lag.

There is one critical element that determines whether this vision actually succeeds in the real world: total user control over data (and i think this was also a part of the task, i put in a memory.control tool in the architecture as well which is specifically to override memory on user command). An assistant that integrates this deeply into someone’s life only works if the user has absolute sovereignty over their own workflows and privacy. People will only rely on a system like this if they retain full ownership and visibility of what the system touches.
