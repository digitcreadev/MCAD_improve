from locust import HttpUser, task, between

class McadUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        # Charger un objectif
        r = self.client.get("/objectives")
        data = r.json()
        self.objective_id = data[0]["id"]

        # Créer une session MCAD
        res = self.client.post("/sessions", json={
            "objective_id": self.objective_id,
            "dw_id": "FOODMART"
        })
        self.session_id = res.json()["session_id"]

    @task
    def evaluate_visual_mdx(self):
        qp = {
            "cube": "Sales",
            "rows": ["Produit.Catégorie"],
            "columns": ["Temps.Mois"],
            "filters": ["Région=Nord", "Année=1998"],
            "measures": ["Marge %", "Rupture %"],
            "force_sat": True,
            "target_constraints": [
                "c_margin_cat_north_1998",
                "c_stockout_cat_north_1998",
                "c_corr_margin_stockout_north_12m"
            ],
            "step_name": "locust_step",
            "step_description": "Evaluation MCAD depuis Locust"
        }

        self.client.post(
            f"/sessions/{self.session_id}/evaluate_visual_mdx",
            json={
                "objective_id": self.objective_id,
                "qp": qp
            }
        )
