# Data Dictionary

## Description_PROC
Overall description of property, location, amenities and key policies.
- [primary key] eg_property_id: Unique property identifier
- guestrating_avg_expedia: Average guest review rating
- city: Location- city
- province: Location- province
- country: Location- country
- star_rating: Hotel star level (1-5) indicating luxury and amenity class
- area_description: Short description of property location
- property_description: Short description of property
- popular_amenities_list: Highlighted amenities
- property_amenity_*: Amenity highlights organized by specific sub-categories	
- check_in_start_time: Earliest check-in time
- check_in_end_time: End of check-in window
- check_out_time: Latest check-out time
- check_out_policy: Additional explanation of check in/out policies
- pet_policy: Pet policy description
- children_and_extra_bed_policy: Policies for children and extra beds
- check_in_instructions: Additional details about check-in process
- know_before_you_go: Additional notes and policies for the property

## Reviews_PROC
Reviews for targeted properties.
- [primary key] eg_property_id: Unique property identifier
- acquisition_date: Date of review submission
- lob: Line of business (inventory type)
- rating: Guest review rating, broken down by sub-category. Rating scale 1-5, with 0 indicating NULL/no rating
- review_title: Guest review title
- review_text: Full review text