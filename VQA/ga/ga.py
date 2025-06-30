import os
import io
import json
import base64
import numpy as np
import cv2
from PIL import Image
import warnings
import time
import argparse 

from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, lab2rgb
from skimage.filters import unsharp_mask
from skimage.restoration import denoise_bilateral

warnings.filterwarnings('ignore')

class OptimizedImageEnhancementGA:
    """Optimized Genetic Algorithm for Image Enhancement"""
    
    def __init__(self, population_size=15, max_generations=12, elite_size=2,
                tournament_size=3, initial_mutation_rate=0.4, initial_crossover_rate=0.8,
                early_stopping_patience=5, target_fitness=0.85):
        """Initialize the optimized genetic algorithm parameters"""
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.initial_mutation_rate = initial_mutation_rate
        self.initial_crossover_rate = initial_crossover_rate
        self.early_stopping_patience = early_stopping_patience
        self.target_fitness = target_fitness
        
        # Adaptive parameters
        self.current_mutation_rate = initial_mutation_rate
        self.current_crossover_rate = initial_crossover_rate
        
        # Cache for fitness calculations
        self.fitness_cache = {}
        
    def safe_image_conversion(self, image):
        """Safely convert image to proper format"""
        try:
            # Ensure image is numpy array
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            
            # Handle different data types
            if image.dtype == np.float64 or image.dtype == np.float32:
                if image.max() <= 1.0 and image.min() >= 0.0:
                    # Already in 0-1 range, good to go
                    image = image.astype(np.float32)
                else:
                    # Convert to 0-1 range
                    image = image.astype(np.float32) / 255.0
                    image = np.clip(image, 0, 1)
            elif image.dtype == np.uint8:
                # Convert uint8 to float32 in 0-1 range
                image = image.astype(np.float32) / 255.0
            
            # Ensure 3 channels
            if len(image.shape) == 2: # Grayscale
                image = np.stack([image, image, image], axis=2)
            elif len(image.shape) == 3 and image.shape[2] == 4: # RGBA
                image = image[:, :, :3]  # Remove alpha channel
            elif len(image.shape) == 3 and image.shape[2] == 1: # Grayscale with 3rd dim
                 image = np.concatenate([image, image, image], axis=2)

            # Final check for NaN or Inf values
            if np.isnan(image).any() or np.isinf(image).any():
                # print("Warning: NaN or Inf values detected in image, replacing with zeros")
                image = np.nan_to_num(image)
                
            # Final range check
            if image.max() > 1.0 or image.min() < 0.0:
                image_min = image.min()
                image_max = image.max()
                if image_max > image_min: # Avoid division by zero
                    image = (image - image_min) / (image_max - image_min)
                else: # Handle flat image
                    image = np.zeros_like(image) 
                image = np.clip(image, 0, 1)
                
            return image.astype(np.float32) # Ensure float32 output
        
        except Exception as e:
            # Create a fallback image
            fallback = np.ones((224, 224, 3), dtype=np.float32) * 0.5
            return fallback
        
    def create_smart_chromosome(self, base_chromosome=None):
        """Create a chromosome with smart initialization"""
        if base_chromosome is None:
            # Smart ranges based on typical good values
            return {
                'gamma': np.random.uniform(0.8, 1.4),  # Narrower, more realistic range
                'contrast': np.random.uniform(0.9, 1.3),
                'saturation': np.random.uniform(0.95, 1.15),
                'color_balance': np.random.uniform(0.95, 1.05, 3),  # Subtle color balance
                'sharpness': np.random.uniform(0.1, 1.0),
                'unsharp_radius': np.random.choice([1.0, 2.0]),  # Simplified options
                'denoise_strength': np.random.uniform(0.0, 0.15),
                'bilateral_sigma_color': np.random.uniform(0.08, 0.15),
                'bilateral_sigma_spatial': np.random.uniform(1.5, 2.5),
                'clahe_clip': np.random.uniform(0.015, 0.03),
                'clahe_kernel': np.random.choice([8, 16]),  # Simplified options
            }
        else:
            # Create variation of existing good chromosome
            return self.mutate_chromosome(base_chromosome, mutation_rate=0.2, mutation_amount=0.15)
        
    def calculate_optimized_fitness(self, original, enhanced):
        """Optimized fitness calculation with fewer but more effective metrics"""
        try:
            # Safely convert both images
            original = self.safe_image_conversion(original)
            enhanced = self.safe_image_conversion(enhanced)
            
            # Create hash for caching
            enhanced_hash = hash(enhanced.tobytes()) # Ensure enhanced is contiguous for tobytes
            if enhanced_hash in self.fitness_cache:
                return self.fitness_cache[enhanced_hash]
            
            # 1. Structural similarity (most important metric)
            try:
                ssim_value = ssim(original, enhanced, data_range=1.0, channel_axis=2, win_size=3, multichannel=True) 
                if np.isnan(ssim_value) or np.isinf(ssim_value):
                    ssim_value = 0.5
            except Exception as ssim_e:
                ssim_value = 0.5 # Fallback if ssim fails
            
            # 2. Simplified contrast measure using standard deviation
            try:
                gray_orig = cv2.cvtColor((original * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                gray_enhanced = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                
                contrast_orig = np.std(gray_orig)
                contrast_enhanced = np.std(gray_enhanced)
                contrast_improvement = min(contrast_enhanced / (contrast_orig + 1e-8), 2.0) / 2.0
            except:
                contrast_improvement = 0.5
            
            # 3. Edge preservation using simplified gradient
            try:
                grad_orig = np.abs(cv2.Laplacian(gray_orig, cv2.CV_32F))
                grad_enhanced = np.abs(cv2.Laplacian(gray_enhanced, cv2.CV_32F))
                
                flat_grad_orig = grad_orig.flatten() if grad_orig.ndim > 1 else grad_orig
                flat_grad_enhanced = grad_enhanced.flatten() if grad_enhanced.ndim > 1 else grad_enhanced

                if len(flat_grad_orig) > 1 and len(flat_grad_enhanced) > 1: # Need at least 2 points for corrcoef
                    corr_coef = np.corrcoef(flat_grad_orig, flat_grad_enhanced)
                    if corr_coef.shape == (2, 2):
                        edge_correlation = corr_coef[0, 1]
                    else:
                        edge_correlation = 0.5
                else:
                    edge_correlation = 0.5 # Not enough data for correlation
                
                if np.isnan(edge_correlation) or np.isinf(edge_correlation):
                    edge_correlation = 0.5
                    
                edge_preservation = max(0, edge_correlation)
            except:
                edge_preservation = 0.5
            
            # 4. Color naturalness - penalize extreme saturation
            try:
                lab_enhanced = rgb2lab(enhanced)
                color_std = np.std(lab_enhanced[:,:,1:])  # a,b channels
                color_naturalness = max(0, 1.0 - (color_std - 20) / 50)  # Optimal around 20
                if np.isnan(color_naturalness) or np.isinf(color_naturalness):
                    color_naturalness = 0.7
            except:
                color_naturalness = 0.7
            
            # 5. Noise assessment - difference in smooth areas
            try:
                smooth_mask = grad_orig < np.percentile(grad_orig, 30)
                if np.sum(smooth_mask) > 100:  # Enough smooth pixels
                    noise_score = 1.0 - np.mean(np.abs(original - enhanced)[smooth_mask]) * 5
                    noise_score = max(0, min(1, noise_score))
                else:
                    noise_score = 0.7
                
                if np.isnan(noise_score) or np.isinf(noise_score):
                    noise_score = 0.7
            except:
                noise_score = 0.7
            
            # Optimized weighted combination
            weights = {
                'ssim': 0.4, 'contrast': 0.25, 'edge': 0.2, 'color': 0.1, 'noise': 0.05
            }
            
            fitness = (
                weights['ssim'] * ssim_value +
                weights['contrast'] * contrast_improvement +
                weights['edge'] * edge_preservation +
                weights['color'] * color_naturalness +
                weights['noise'] * noise_score
            )
            
            if np.isnan(fitness) or np.isinf(fitness): fitness = 0.3
            
            metrics = {
                'fitness': fitness, 'ssim': ssim_value, 'contrast': contrast_improvement,
                'edge': edge_preservation, 'color': color_naturalness, 'noise': noise_score
            }
            
            self.fitness_cache[enhanced_hash] = metrics
            return metrics
            
        except Exception as e:
            default_metrics = {
                'fitness': 0.3, 'ssim': 0.5, 'contrast': 0.5, 'edge': 0.5, 'color': 0.5, 'noise': 0.5
            }
            return default_metrics
    
    def enhance_image_optimized(self, image, chromosome):
        """Optimized image enhancement with fewer conversions"""
        try:
            img = self.safe_image_conversion(image.copy())
            for i in range(3): img[:,:,i] *= chromosome['color_balance'][i]
            img = np.clip(img, 0, 1)
            img = np.power(img, 1.0/chromosome['gamma'])
            img = np.clip(img, 0, 1)
            img = np.clip((img - 0.5) * chromosome['contrast'] + 0.5, 0, 1)
            if abs(chromosome['saturation'] - 1.0) > 1e-3:
                try:
                    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                    hsv[:,:,1] *= chromosome['saturation']
                    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 1)
                    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    img = np.clip(img, 0, 1)
                except Exception as sat_e: pass
            if chromosome['sharpness'] > 0.05:
                try:
                    img_sharp = unsharp_mask(img, radius=chromosome['unsharp_radius'], 
                                     amount=chromosome['sharpness'], channel_axis=-1, preserve_range=True)
                    img = np.clip(img_sharp, 0, 1)
                except Exception as sharp_e: pass
            if chromosome['denoise_strength'] > 0.01:
                try:
                    img_denoised = denoise_bilateral(img, 
                                          sigma_color=chromosome['bilateral_sigma_color'],
                                          sigma_spatial=chromosome['bilateral_sigma_spatial'],
                                          channel_axis=-1, preserve_range=True)
                    img = np.clip(img_denoised, 0, 1)
                except Exception as denoise_e: pass
            if chromosome['clahe_clip'] > 0.005:
                try:
                    lab_img = rgb2lab(img)
                    l_channel = lab_img[:,:,0]
                    l_normalized = np.clip(l_channel / 100.0 * 255.0, 0, 255).astype(np.uint8)
                    clahe_kernel_size = int(chromosome['clahe_kernel'])
                    clahe = cv2.createCLAHE(clipLimit=chromosome['clahe_clip']*100,
                                           tileGridSize=(clahe_kernel_size, clahe_kernel_size))
                    l_clahe = clahe.apply(l_normalized)
                    lab_img[:,:,0] = l_clahe.astype(np.float32) / 255.0 * 100.0
                    img_clahe = lab2rgb(lab_img)
                    img = np.clip(img_clahe, 0, 1)
                except Exception as clahe_e: pass
            
            img = np.clip(img, 0, 1)
            if np.isnan(img).any() or np.isinf(img).any():
                img = np.nan_to_num(img)
                img = np.clip(img, 0, 1)
            
            return img.astype(np.float32)
            
        except Exception as e:
            return self.safe_image_conversion(image)
    
    def mutate_chromosome(self, chromosome, mutation_rate=None, mutation_amount=0.15):
        if mutation_rate is None: mutation_rate = self.current_mutation_rate
        mutated = chromosome.copy()
        mutation_applied = False
        for key in mutated:
            if np.random.random() < mutation_rate:
                mutation_applied = True
                if isinstance(mutated[key], np.ndarray):
                    noise = np.random.uniform(1 - mutation_amount, 1 + mutation_amount, mutated[key].shape)
                    mutated[key] *= noise
                    mutated[key] = np.clip(mutated[key], 0.8, 1.2)
                elif key in ['unsharp_radius', 'clahe_kernel']:
                    if key == 'unsharp_radius': mutated[key] = np.random.choice([1.0, 2.0])
                    else: mutated[key] = np.random.choice([8, 16])
                else:
                    current_val = mutated[key]
                    change = np.random.uniform(1 - mutation_amount, 1 + mutation_amount)
                    mutated[key] = current_val * change
                    if key == 'gamma': mutated[key] = np.clip(mutated[key], 0.7, 1.5)
                    elif key == 'contrast': mutated[key] = np.clip(mutated[key], 0.8, 1.4)
                    elif key == 'saturation': mutated[key] = np.clip(mutated[key], 0.9, 1.2)
                    elif key == 'sharpness': mutated[key] = np.clip(mutated[key], 0.0, 1.2)
                    elif key == 'denoise_strength': mutated[key] = np.clip(mutated[key], 0.0, 0.2)
                    elif key == 'bilateral_sigma_color': mutated[key] = np.clip(mutated[key], 0.05, 0.2)
                    elif key == 'bilateral_sigma_spatial': mutated[key] = np.clip(mutated[key], 1.0, 3.0)
                    elif key == 'clahe_clip': mutated[key] = np.clip(mutated[key], 0.01, 0.035)
        if not mutation_applied and np.random.random() < 0.5:
            key_to_mutate = np.random.choice(list(mutated.keys()))
            if key_to_mutate not in ['unsharp_radius', 'clahe_kernel'] and not isinstance(mutated[key_to_mutate], np.ndarray):
                mutated[key_to_mutate] *= np.random.uniform(0.98, 1.02)
                if key_to_mutate == 'gamma': mutated[key_to_mutate] = np.clip(mutated[key_to_mutate], 0.7, 1.5)
        return mutated
        
    def crossover(self, parent1, parent2):
        if np.random.random() > self.current_crossover_rate:
            return parent1.copy() if np.random.random() < 0.5 else parent2.copy()
        child = {}
        for key in parent1:
            if np.random.random() < 0.5:
                child[key] = parent1[key].copy() if isinstance(parent1[key], np.ndarray) else parent1[key]
            else:
                child[key] = parent2[key].copy() if isinstance(parent2[key], np.ndarray) else parent2[key]
        return child
        
    def tournament_selection(self, population, fitness_scores):
        actual_tournament_size = min(self.tournament_size, len(population))
        if actual_tournament_size == 0:
             return population[0] if population else self.create_smart_chromosome()
        tournament_indices = np.random.choice(len(population), actual_tournament_size, replace=False)
        tournament_fitness_values = []
        for i in tournament_indices:
            score = fitness_scores[i]
            if isinstance(score, dict) and 'fitness' in score:
                tournament_fitness_values.append(score['fitness'])
            elif isinstance(score, (float, int)):
                tournament_fitness_values.append(score)
            else:
                tournament_fitness_values.append(0.0)
        winner_idx_in_tournament = np.argmax(tournament_fitness_values)
        winner_original_idx = tournament_indices[winner_idx_in_tournament]
        return population[winner_original_idx]
    
    def evaluate_individual(self, chromosome, original_image):
        try:
            enhanced = self.enhance_image_optimized(original_image, chromosome)
            fitness_metrics = self.calculate_optimized_fitness(original_image, enhanced)
            return enhanced, fitness_metrics
        except Exception as e:
            original_safe = self.safe_image_conversion(original_image)
            default_fitness = {
                'fitness': 0.1, 'ssim': 0.1, 'contrast': 0.1,
                'edge': 0.1, 'color': 0.1, 'noise': 0.1
            }
            return original_safe, default_fitness
    
    def adapt_parameters(self, generation, no_improvement_count):
        if no_improvement_count > self.early_stopping_patience / 2:
            self.current_mutation_rate = min(0.7, self.initial_mutation_rate * (1 + 0.1 * no_improvement_count) )
        else:
            self.current_mutation_rate = self.initial_mutation_rate
        progress_factor = (self.max_generations - generation) / self.max_generations
        if no_improvement_count > 2:
            self.current_crossover_rate = min(0.95, self.initial_crossover_rate * 1.1)
        else:
            self.current_crossover_rate = self.initial_crossover_rate * (0.8 + 0.2 * progress_factor)
        self.current_crossover_rate = np.clip(self.current_crossover_rate, 0.5, 0.95)
    
    def run(self, original_image_input):
        start_time = time.time()
        try:
            original_image_processed = self.safe_image_conversion(original_image_input.copy())
            population = []
            num_base_variations = self.population_size // 3
            num_random = self.population_size - (2 * num_base_variations)
            base_chromosomes = [
                {'gamma': 1.0, 'contrast': 1.1, 'saturation': 1.0, 'color_balance': np.array([1.0, 1.0, 1.0]),
                 'sharpness': 0.3, 'unsharp_radius': 1.0, 'denoise_strength': 0.05, 
                 'bilateral_sigma_color': 0.1, 'bilateral_sigma_spatial': 2.0,
                 'clahe_clip': 0.02, 'clahe_kernel': 8},
                {'gamma': 0.95, 'contrast': 1.15, 'saturation': 1.0, 'color_balance': np.array([1.0, 1.0, 1.0]),
                 'sharpness': 0.4, 'unsharp_radius': 1.0, 'denoise_strength': 0.05,
                 'bilateral_sigma_color': 0.1, 'bilateral_sigma_spatial': 1.5,
                 'clahe_clip': 0.02, 'clahe_kernel': 8}
            ]
            for base in base_chromosomes:
                for _ in range(num_base_variations):
                    if len(population) < self.population_size:
                        population.append(self.create_smart_chromosome(base_chromosome=base))
            for _ in range(num_random):
                 if len(population) < self.population_size:
                    population.append(self.create_smart_chromosome())
            while len(population) < self.population_size:
                population.append(self.create_smart_chromosome())
            best_chromosome_overall = None
            best_fitness_overall = -1.0
            best_enhanced_overall = None
            history = []
            no_improvement_count = 0
            last_best_fitness_val = -1.0
            
            for generation in range(self.max_generations):
                gen_start_time = time.time()
                enhanced_images_current_gen = []
                fitness_scores_current_gen = []
                for chrom in population:
                    enhanced, fitness_dict = self.evaluate_individual(chrom, original_image_processed)
                    enhanced_images_current_gen.append(enhanced)
                    fitness_scores_current_gen.append(fitness_dict)
                current_gen_best_fitness_val = -1.0
                current_gen_best_idx = -1
                for i, f_dict in enumerate(fitness_scores_current_gen):
                    if f_dict['fitness'] > current_gen_best_fitness_val:
                        current_gen_best_fitness_val = f_dict['fitness']
                        current_gen_best_idx = i
                if current_gen_best_fitness_val > best_fitness_overall:
                    best_fitness_overall = current_gen_best_fitness_val
                    best_chromosome_overall = population[current_gen_best_idx].copy()
                    best_enhanced_overall = enhanced_images_current_gen[current_gen_best_idx]
                if best_fitness_overall > last_best_fitness_val + 0.001:
                    no_improvement_count = 0
                    last_best_fitness_val = best_fitness_overall
                else:
                    no_improvement_count += 1
                self.adapt_parameters(generation, no_improvement_count)
                avg_fitness_val_this_gen = np.mean([f['fitness'] for f in fitness_scores_current_gen])
                gen_stats = {
                    'generation': generation, 'best_fitness': best_fitness_overall,
                    'avg_fitness': avg_fitness_val_this_gen, 'mutation_rate': self.current_mutation_rate,
                    'crossover_rate': self.current_crossover_rate, 'no_improvement': no_improvement_count,
                    'time': time.time() - gen_start_time
                }
                history.append(gen_stats)
                if best_fitness_overall >= self.target_fitness: break
                if no_improvement_count >= self.early_stopping_patience: break
                
                new_population = []
                elite_indices = sorted(range(len(fitness_scores_current_gen)), key=lambda k: fitness_scores_current_gen[k]['fitness'], reverse=True)[:self.elite_size]
                for idx in elite_indices:
                    new_population.append(population[idx])
                
                while len(new_population) < self.population_size:
                    parent1 = self.tournament_selection(population, fitness_scores_current_gen)
                    parent2 = self.tournament_selection(population, fitness_scores_current_gen)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate_chromosome(child)
                    new_population.append(child)
                population = new_population
            
            if best_enhanced_overall is None:
                best_enhanced_overall = original_image_processed
                best_chromosome_overall = self.create_smart_chromosome()
                best_fitness_overall = 0.1

            if np.isnan(best_enhanced_overall).any() or np.isinf(best_enhanced_overall).any():
                best_enhanced_overall = np.nan_to_num(best_enhanced_overall)
                best_enhanced_overall = np.clip(best_enhanced_overall, 0, 1)
                if np.isnan(best_enhanced_overall).any() or np.isinf(best_enhanced_overall).any():
                    best_enhanced_overall = original_image_processed
                    best_fitness_overall = 0.1
            
            best_enhanced_overall = np.clip(best_enhanced_overall, 0, 1).astype(np.float32)
            
            return best_enhanced_overall, best_chromosome_overall, best_fitness_overall, history
            
        except Exception as e:
            original_safe = self.safe_image_conversion(original_image_input)
            default_chromosome = self.create_smart_chromosome()
            return original_safe, default_chromosome, 0.1, []

def initialize_ga(args):
    """Initializes the GA with parameters from command-line arguments."""
    return OptimizedImageEnhancementGA(
        population_size=args.population_size,
        max_generations=args.max_generations,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        initial_mutation_rate=args.mutation_rate,
        initial_crossover_rate=args.crossover_rate,
        early_stopping_patience=args.patience,
        target_fitness=args.target_fitness
    )

# ... (Hàm robust_image_conversion và enhance_image_safe giữ nguyên) ...
def robust_image_conversion(image_array_input):
    """Ultra-robust image conversion that handles all edge cases, aims for uint8 RGB output"""
    try:
        image_array = image_array_input.copy()
        if not isinstance(image_array, np.ndarray): image_array = np.array(image_array)
        if len(image_array.shape) == 2: image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif len(image_array.shape) == 3:
            if image_array.shape[2] == 1: image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4: image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            elif image_array.shape[2] != 3:
                if image_array.shape[2] > 3: image_array = image_array[:,:,:3]
                else: raise ValueError(f"Unsupported number of channels: {image_array.shape[2]}")
        
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0 and image_array.min() >= -1.0:
                if image_array.min() < 0: image_array = ((image_array + 1.0) / 2.0 * 255.0)
                else: image_array = (image_array * 255.0)
            elif image_array.max() > 255.0 or image_array.min() < 0:
                image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        if image_array.shape[2] != 3:
            height, width = image_array.shape[:2] if len(image_array.shape) >=2 else (224,224)
            return np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        return image_array
        
    except Exception as e:
        h, w = (224, 224)
        if isinstance(image_array_input, np.ndarray) and image_array_input.ndim >=2:
            h,w = image_array_input.shape[:2]
        return np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)

def enhance_image_safe(image_bgr_input, ga_instance):
    """Enhanced image enhancement, now accepts a GA instance."""
    try:
        original_rgb_uint8 = cv2.cvtColor(image_bgr_input, cv2.COLOR_BGR2RGB)
        try:
            enhanced_image_float_rgb, best_params, fitness_scalar, history = ga_instance.run(original_rgb_uint8)
            if enhanced_image_float_rgb.max() <= 1.0 and enhanced_image_float_rgb.min() >=0.0 :
                enhanced_uint8_rgb = (enhanced_image_float_rgb * 255.0).astype(np.uint8)
            else:
                enhanced_uint8_rgb = robust_image_conversion(enhanced_image_float_rgb)

            if enhanced_uint8_rgb.mean() < 10 or np.array_equal(enhanced_uint8_rgb, np.zeros_like(enhanced_uint8_rgb)):
                enhanced_uint8_rgb = original_rgb_uint8.copy()
                best_params = { 'gamma': 1.0, 'contrast': 1.0, 'saturation': 1.0, 'sharpness': 0.0, 'denoise_strength': 0.0 }
                fitness_scalar = 0.0
            
        except Exception as ga_error:
            enhanced_uint8_rgb = original_rgb_uint8.copy()
            best_params = { 'gamma': 1.0, 'contrast': 1.0, 'saturation': 1.0, 'sharpness': 0.0, 'denoise_strength': 0.0 }
            fitness_scalar = 0.0
        
        enhancement_info = {
            "gamma": f"{best_params.get('gamma', 1.0):.2f}", "contrast": f"{best_params.get('contrast', 1.0):.2f}",
            "saturation": f"{best_params.get('saturation', 1.0):.2f}", "sharpness": f"{best_params.get('sharpness', 0.0):.2f}",
            "denoise_strength": f"{best_params.get('denoise_strength', 0.0):.2f}", "fitness_score": f"{fitness_scalar:.3f}"
        }
        return enhanced_uint8_rgb, enhancement_info
        
    except Exception as e:
        fallback_img_rgb_uint8 = robust_image_conversion(cv2.cvtColor(image_bgr_input, cv2.COLOR_BGR2RGB))
        fallback_info = {
            "gamma": "1.00", "contrast": "1.00", "saturation": "1.00",
            "sharpness": "0.00", "denoise_strength": "0.00", "fitness_score": "0.00"
        }
        return fallback_img_rgb_uint8, fallback_info


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Batch Image Enhancement using Genetic Algorithm.")
    
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Path to the directory containing input images.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Path to the directory to save enhanced images.")
    
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit the number of images to process. -1 means all images. (Default: -1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files in the output directory.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output during processing.")
                        
    parser.add_argument("--population_size", type=int, default=4, help="GA population size (Default: 4)")
    parser.add_argument("--max_generations", type=int, default=5, help="GA max generations (Default: 5)")
    parser.add_argument("--elite_size", type=int, default=2, help="GA elite size (Default: 2)")
    parser.add_argument("--tournament_size", type=int, default=3, help="GA tournament size (Default: 3)")
    parser.add_argument("--mutation_rate", type=float, default=0.3, help="GA initial mutation rate (Default: 0.3)")
    parser.add_argument("--crossover_rate", type=float, default=0.7, help="GA initial crossover rate (Default: 0.7)")
    parser.add_argument("--patience", type=int, default=3, help="GA early stopping patience (Default: 3)")
    parser.add_argument("--target_fitness", type=float, default=0.55, help="GA target fitness (Default: 0.55)")
    
    return parser.parse_args()

def main():
    """Main function to run the batch processing."""
    args = parse_arguments()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    ga = initialize_ga(args)

    allowed_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    try:
        image_filenames = [f for f in os.listdir(input_dir) if f.lower().endswith(allowed_extensions)]
    except FileNotFoundError:
        print(f"Error: Input directory '{input_dir}' not found.")
        return 

    if args.limit > 0:
        image_filenames = image_filenames[:args.limit]
        print(f"Processing a limit of {args.limit} images.")

    print(f"Found {len(image_filenames)} images to process in '{input_dir}'.")
    print(f"Output will be saved to '{output_dir}'.")

    for i, filename in enumerate(image_filenames):
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename)
    
        if args.verbose:
            print(f"\nProcessing image {i+1}/{len(image_filenames)}: {filename}")
    
        if not args.overwrite and os.path.exists(output_image_path):
            if args.verbose:
                print(f"Output file {output_image_path} already exists. Skipping.")
            continue
            
        try:
            image_bgr = cv2.imread(input_image_path)
    
            if image_bgr is None:
                print(f"Warning: Could not read image {input_image_path}. Skipping.")
                continue
            
            enhanced_image_rgb, enhancement_info = enhance_image_safe(image_bgr, ga)
            
            enhanced_image_bgr_to_save = cv2.cvtColor(enhanced_image_rgb, cv2.COLOR_RGB2BGR)
            
            success = cv2.imwrite(output_image_path, enhanced_image_bgr_to_save)
            
            if success:
                if args.verbose:
                    print(f"Successfully enhanced and saved to {output_image_path}")
                    print(f"Enhancement Info: {enhancement_info}")
            else:
                print(f"Error: Failed to save enhanced image {output_image_path}")
    
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            print("Skipping to next image.")
            continue

    print("\nBatch image enhancement process complete.")

if __name__ == "__main__":
    main()
