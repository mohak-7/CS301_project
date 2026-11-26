### Case studies (selected by a fixed rule, §7.2)

#### Success (spider, bm25-G1 vs. bm25)

Question: What are the different models for the cards produced after 1980?
Gold: ⟨car_1, {model_list, car_names, cars_data}⟩
- bm25 top 5: car_1.model_list, customers_card_transactions.customers_cards, imdb.producer, phone_1.chip_model, phone_1.phone
- bm25-G1 top 5: car_1.model_list, customers_card_transactions.customers_cards, car_1.car_names, phone_1.chip_model, phone_1.phone

#### Failure (spider, bm25-G1 vs. bm25)

Question: How many documents do we have?
Gold: ⟨cre_Doc_Template_Mgt, {documents}⟩
- bm25 top 5: document_management.documents, document_management.document_structures, cre_Docs_and_Epenses.documents, cre_Doc_Tracking_DB.all_documents, cre_Doc_Template_Mgt.documents
- bm25-G1 top 5: document_management.documents, document_management.document_structures, cre_Docs_and_Epenses.documents, document_management.document_functional_areas, cre_Doc_Tracking_DB.all_documents

#### Success (bird, bm25-G1 vs. bm25)

Question: Please list the codes of the schools with a total enrollment of over 500.
Gold: ⟨california_schools, {frpm, schools}⟩
- bm25 top 5: student_loan.enrolled, california_schools.satscores, california_schools.frpm, movie_platform.lists, donor.projects
- bm25-G1 top 5: student_loan.enrolled, california_schools.satscores, california_schools.frpm, movie_platform.lists, california_schools.schools

#### Failure (bird, bm25-G1 vs. bm25)

Question: Please list the websites where I can purchase the cards that have the promotional type of "bundle".
Gold: ⟨card_games, {cards}⟩
- bm25 top 5: beer_factory.transaction, movie_platform.lists, financial.card, works_cycles.creditcard, card_games.cards
- bm25-G1 top 5: beer_factory.transaction, movie_platform.lists, beer_factory.customers, beer_factory.rootbeer, movie_platform.lists_users

#### Success (spider, sxfmr-G2 vs. sxfmr)

Question: Which city has the most frequent destination airport?
Gold: ⟨flight_2, {airports, flights}⟩
- sxfmr top 5: flight_2.airports, aircraft.airport, flight_4.airports, flight_company.airport, geo.city
- sxfmr-G2 top 5: flight_2.airports, flight_2.flights, aircraft.airport, flight_4.airports, flight_company.airport

#### Failure (spider, sxfmr-G2 vs. sxfmr)

Question: What are all distinct countries where singers above age 20 are from?
Gold: ⟨concert_singer, {singer}⟩
- sxfmr top 5: singer.singer, music_1.artist, concert_singer.singer, music_1.genre, singer.song
- sxfmr-G2 top 5: singer.singer, singer.song, music_1.artist, music_1.genre, music_1.song
