#!/home/picocluster/k3s_monitor_env/venv/bin/python3
import os
from kubernetes import client, config, watch
from prometheus_client import start_http_server, Gauge, REGISTRY, PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR

# Optimisation : Suppression des métriques Python inutiles pour sauver la RAM
REGISTRY.unregister(PROCESS_COLLECTOR)
REGISTRY.unregister(PLATFORM_COLLECTOR)
# REGISTRY.unregister(REGISTRY._names_to_collectors['python_gc_objects_collected_total', 'python_gc_objects_uncollectable_total', 'python_gc_collections_total'])

# 2. Désactiver spécifiquement le collecteur Garbage Collector (GC)
# C'est la méthode officielle pour supprimer les métriques python_gc_*
try:
    REGISTRY.unregister(GC_COLLECTOR)
except Exception:
    pass

# 3. Optionnel : Nettoyage manuel si certaines persistent
for name in list(REGISTRY._names_to_collectors.keys()):
    if name.startswith('python_'):
        try:
            REGISTRY.unregister(REGISTRY._names_to_collectors[name])
        except KeyError:
            pass

# metric structure: k3s_job_status{job_name="myjob", namespace="default"} 1
# usage: k3s_job_status == 1 (succ), 0 (fail), 2 (running) <-- permet de faire des alertes sur les échecs ou les jobs qui tournent trop longtemps
# usage: k3s_job_status.labels.job_name <-- retourne le nom du job
# usage: k3s_job_status.labels.namespace <-- retourne le namespace du job
job_status = Gauge('k3s_job_status', 'Status of K3s Job (1=Succ, 0=Fail, 2=Run)', ['job_name', 'namespace'])

# add new metric for pods, then implement later
# pod_status = Gauge('k3s_pod_status', 'Status of K3s Pod (1=Running, 0=Not Running)', ['pod_name', 'namespace'])

def monitor():
    # Utilise ~/.kube/config, <-- check /etc/systemd/system/k3s-job-watcher.service KUBECONFIG env var
    config.load_kube_config()
    v1 = client.BatchV1Api()
    w = watch.Watch()
    
    # Le stream est passif (consomme peu de CPU)
    for event in w.stream(v1.list_job_for_all_namespaces):
        job = event['object']
        name, ns = job.metadata.name, job.metadata.namespace
        if job.status.succeeded:
            job_status.labels(job_name=name, namespace=ns).set(1)
        elif job.status.failed:
            job_status.labels(job_name=name, namespace=ns).set(0)
        else:
            job_status.labels(job_name=name, namespace=ns).set(2)

if __name__ == '__main__':
    start_http_server(9101)
    monitor()
