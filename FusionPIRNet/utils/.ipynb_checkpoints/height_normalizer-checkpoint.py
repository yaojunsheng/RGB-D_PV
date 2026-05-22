# utils/height_normalizer.py
class HeightNormalizer:
    def __init__(self, p5=-0.026, p95=10.206):
        self.p5 = p5
        self.p95 = p95
    
    def normalize(self, height):
        """归一化高度数据到[0,1]"""
        height = torch.clamp(height, self.p5, self.p95)
        return (height - self.p5) / (self.p95 - self.p5)
    
    def denormalize(self, normalized):
        """反归一化用于评估"""
        return normalized * (self.p95 - self.p5) + self.p5