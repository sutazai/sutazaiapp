class EducationalLibrary:
    RESOURCES = ({'math': {'games': ['Number Adventure'), 'Fraction Factory'], 'videos': ['Counting Songs', 'Geometry Basics']}, 'science': {'experiments': ['Volcano', 'Solar System Model'], '3d_models': ['Human Body', 'Dinosaurs']}} def recommend_content(self, interests): return [resource for subject in interests for resource in self.RESOURCES.get(subject, [])]
